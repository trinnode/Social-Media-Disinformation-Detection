"""
Social Media Disinformation Detection System
Main entry point for training, evaluation, and application launch.
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(
        description="Social Media Disinformation Detection System"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "app", "generate-data", "all"],
        default="all",
        help="Mode of operation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit app"
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate-data" or args.mode == "all":
        print("\n" + "="*60)
        print("  Generating Dataset")
        print("="*60)
        from data.generate_dataset import generate_dataset
        generate_dataset()
    
    if args.mode == "train" or args.mode == "all":
        print("\n" + "="*60)
        print("  Training Models")
        print("="*60)
        from train import Trainer
        trainer = Trainer(device=args.device)
        trainer.train_all()
    
    if args.mode == "evaluate" or args.mode == "all":
        print("\n" + "="*60)
        print("  Evaluating Models")
        print("="*60)
        from evaluate import Evaluator
        evaluator = Evaluator(device=args.device)
        evaluator.run_full_evaluation()
    
    if args.mode == "app":
        print("\n" + "="*60)
        print("  Launching Web Application")
        print("="*60)
        os.system(f"streamlit run app/streamlit_app.py --server.port {args.port}")


if __name__ == "__main__":
    main()
