import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, glue_compute_metrics, glue_output_modes, glue_tasks_num_labels, set_seed

import arguments
from modeling_san import SANForSequenceClassification
from trainer import Trainer, TrainingArguments


# set up framework to allow paths disentangledment if
# specified as such in the command line.
do_paths = True
# specifying number of random labels.
# 0 means disable randmization.
random_labels = 2

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # parser.add_argument('n_gpu', default=1)
    """
        if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")                                              args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()                                                                       else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs                                                torch.cuda.set_device(args.local_rank)                                                                                                  device = torch.device("cuda", args.local_rank)                                                                                          torch.distributed.init_process_group(backend="nccl")                                                                                    args.n_gpu = 1                                                                                                                      args.device = device
    """
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        try:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        except ValueError as e:
            print('Additional arguments passed not parsed by ArgumentParser: {}'.format(e))

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    args = arguments.parse_args()
    # Set seed
    set_seed(training_args.seed if args.seed is None else args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name] if random_labels == 0 else random_labels
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        hidden_act="gelu",
        intermediate_size=args.hidden_dim,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    config.max_position_embeddings = -1

    config.train_loss_l = []
    # Can enable test_width to observe effects of widths.
    test_width = False
    if test_width:
        depth = True
        depth = False

        if depth:
            print("DEPTH")
            config.num_hidden_layers = 3  # 8 #8
            config.num_attention_heads = 3  # 8
            dim_per_head = 51
        else:
            print("BREADTH")
            config.num_hidden_layers = 6  # 5
            config.num_attention_heads = 1  # 1 #10 6
            dim_per_head = 64
        print(config.num_hidden_layers, config.num_attention_heads)
        config.hidden_size = config.num_attention_heads * dim_per_head
        config.intermediate_size = 2 * config.hidden_size

    if do_paths:
        config.num_hidden_layers = args.depth  # 1
        config.num_attention_heads = args.width  # 1
        config.hidden_dropout_prob = 0.1
        config.hidden_size = args.hidden_dim
        config.intermediate_size = config.hidden_size  # *2
        # hidden_size is total hidden dimension, to be divided across heads, num_heads is number of heads

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if do_paths or test_width:
        # model = AutoModelForSequenceClassification.from_config(config=config)
        model = SANForSequenceClassification(config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    if do_paths or test_width:
        # count params
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for name, p in model.named_parameters():
            print("p {} size {}".format(name, p.numel()))
        print("+++TOTAL model params {}+++".format(total_params))

    model_path = "modelNLP{}d{}_{}_{}.pt".format(
        config.num_attention_heads, config.num_hidden_layers, config.hidden_size, int(training_args.num_train_epochs)
    )  # modelNLP4d6_256_10.pt
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(training_args.device)
        training_args.num_train_epochs = 0

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids) if random_labels == 0 else 0

        return compute_metrics_fn

    if random_labels:
        print("Note: random labels assigned to data!")
        target_feat_len = min(len(train_dataset.features), args.n_train_data)
        train_dataset.features = train_dataset.features[:target_feat_len]

        # traindataset consists of InputFeatures objects, each having label, attention mask, input_ids, etc.
        # torch.manual_seed(44)
        rand = torch.randint(random_labels, (len(train_dataset),))
        n_tok = len(train_dataset[0].input_ids)
        # Learn and evaluate at token-level.
        test_token = True
        for i, feat in enumerate(train_dataset):
            if test_token:
                rand_labels = torch.randint(random_labels, (n_tok,))  # len(feat.input_ids)
                rand_labels[torch.Tensor(feat.attention_mask) == 0] = -1
                feat.label = rand_labels
            else:
                feat.label = rand[i]

    training_args.logging_steps = 100
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        args_sort=args,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        ##eval_datasets = [eval_dataset]
        eval_datasets = [train_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        # config.no_sub_path = args.no_sub_path
        # generate paths
        # convex_hull.create_path(path_len, args, all_heads=args.all_heads )
        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )

            # eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    print("ALL EPOCH LOSSES {}".format(config.train_loss_l))
    return eval_results


if __name__ == "__main__":
    main()
