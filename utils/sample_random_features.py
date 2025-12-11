import argparse
import random
from sae_lens import SAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_release", required=True)
    parser.add_argument("--sae_id", required=True)
    parser.add_argument("--k", type=int, default=20,
                        help="How many random features to sample")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    # Load SAE 
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device="cpu",    
    )

    # Number of features in this SAE
    if hasattr(sae, "W_dec"):
        n_features = sae.W_dec.shape[0]
    else:
        n_features = sae.cfg.d_sae

    if args.k > n_features:
        raise ValueError(f"Requested {args.k} features, but SAE only has {n_features}")

    random.seed(args.seed)
    feats = sorted(random.sample(range(n_features), args.k))

    # Print comma-separated list
    print(",".join(str(f) for f in feats))

if __name__ == "__main__":
    main()
