import torch
import jax.numpy as jnp
import jax
import random

def compute_low_rank(x, k=1):
    U, s, Vh = jax.vmap(jnp.linalg.svd)(x)
    return jnp.einsum("ij,j,jk->ik", U[:, :k], s[:k], Vh[:k ,:])

def l1_matrix_norm(x):
    return x.abs().sum(axis=-2 % x.ndim).max(axis=-1).values

def linf_matrix_norm(x):
    return l1_matrix_norm(x.transpose(-2, -1))

def composite_norm(x):
    return torch.sqrt(l1_matrix_norm(x) * linf_matrix_norm(x))

all_norms = {
    "l1": l1_matrix_norm,
    "l2": lambda r: torch.norm(r, p=2, dim=(-2, -1)),
    "l_inf": linf_matrix_norm,
    "l1 * l_inf": composite_norm,
}

all_norms_names = list(all_norms.keys())

def sample_path(depth, num_layers, num_heads):
    selected_layers = sorted(random.sample(list(range(num_layers)), depth))
    selected_heads = random.choices(list(range(num_heads)), k=depth)
    return selected_layers, selected_heads

def sample_P_matrix(attentions, depth: int):
    num_layers, num_samples, num_heads, t, _ = attentions.shape
    selected_layers, selected_heads = sample_path(depth, num_layers, num_heads)
    sample_idx = random.choice(list(range(num_samples)))
    P = torch.eye(t)
    for layer, head in zip(selected_layers, selected_heads):
        P = P @ attentions[layer, sample_idx, head]
    return P
