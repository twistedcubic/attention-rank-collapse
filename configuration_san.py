from transformers.configuration_bert import BertConfig


class SANConfig(BertConfig):
    def __init__(self, do_mlp=False, do_ffn2_embed=False, no_embed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_mlp = do_mlp
        self.do_ffn2_embed = do_ffn2_embed
        self.no_embed = no_embed
