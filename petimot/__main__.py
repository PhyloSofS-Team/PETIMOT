import typer

app = typer.Typer()


@app.command(name="infer")
def run_infer(
    model_path: str = typer.Option(
        "",
        "--model-path",
        "-m",
        help="Path to model checkpoint",
    ),
    config_file: str = typer.Option(
        "configs/default.yaml",
        "--config-file",
        "-c",
        help="Path to configuration file",
    ),
    list_path: str = typer.Option(
        "",
        "--list-path",
        "-l",
        help="Path to file containing input files or sample indices, or direct path to a PDB/PT file",
    ),
    output_path: str = typer.Option(
        "predictions",
        "--output-path",
        "-o",
        help="Output directory path",
    ),
):
    """
    Run inference using a pretrained model. The input can be:
    1. A direct path to a .pdb or .pt file
    2. A text file containing paths or sample names
    """
    from petimot.infer.infer import infer

    if list_path.endswith((".pdb", ".pt")):
        input_list = [list_path]
    else:
        try:
            with open(list_path, "r") as f:
                input_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise typer.BadParameter(f"Error reading list file: {e}")

    if not input_list:
        raise typer.BadParameter("No input files specified")

    infer(
        model_path=model_path,
        config_file=config_file,
        input_list=input_list,
        output_path=output_path,
    )


@app.command(name="evaluate")
def run_evaluate(
    prediction_path: str = typer.Option(
        "",
        "--prediction-path",
        "-p",
        help="Path to the directory containing predicted modes.",
    ),
    sample_ids_file: str = typer.Option(
        "eval_list.txt",
        "--list-path",
        "-l",
        help="Path to a text file containing sample IDs (one per line).",
    ),
    ground_truth_path: str = typer.Option(
        "ground_truth",
        "--ground-truth-path",
        "-g",
        help="Path to the directory containing ground truth .pt files.",
    ),
    output_path: str = typer.Option(
        "evaluation",
        "--output-path",
        "-o",
        help="Path to the directory where evaluation results will be saved.",
    ),
    num_modes_pred: int = typer.Option(
        4,
        "--num-modes-pred",
        "-np",
        help="Number of predicted modes.",
    ),
    num_modes_gt: int = typer.Option(
        4,
        "--num-modes-gt",
        "-ng",
        help="Number of ground truth modes.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Device to use for evaluation, e.g., 'cuda' or 'cpu'.",
    ),
):
    """
    Evaluate predicted modes against ground truth data stored in .pt files.
    """
    from petimot.eval.eval import evaluate

    sample_ids = None
    if sample_ids_file:
        try:
            with open(sample_ids_file, "r") as f:
                sample_ids = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(sample_ids)} sample IDs from {sample_ids_file}")
        except Exception as e:
            raise typer.BadParameter(f"Error reading sample IDs file: {e}")

    evaluate(
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        sample_ids=sample_ids,
        num_modes_pred=num_modes_pred,
        num_modes_gt=num_modes_gt,
        device=device,
    )


@app.command(name="infer_and_evaluate")
def run_infer_and_evaluate(
    model_path: str = typer.Option(
        "",
        "--model-path",
        "-m",
        help="Path to model checkpoint",
    ),
    config_file: str = typer.Option(
        "configs/default.yaml",
        "--config-file",
        "-c",
        help="Path to configuration file",
    ),
    list_path: str = typer.Option(
        "eval_list.txt",
        "--list-path",
        "-l",
        help="Path to file containing input files or sample indices",
    ),
    ground_truth_path: str = typer.Option(
        "ground_truth",
        "--ground-truth-path",
        "-g",
        help="Path to the directory containing ground truth .pt files.",
    ),
    prediction_path: str = typer.Option(
        "predictions",
        "--prediction-path",
        "-p",
        help="Directory path for saving predictions",
    ),
    evaluation_path: str = typer.Option(
        "evaluation",
        "--evaluation-path",
        "-e",
        help="Directory path for saving evaluation results",
    ),
    num_modes_pred: int = typer.Option(
        4,
        "--num-modes-pred",
        "-np",
        help="Number of predicted modes.",
    ),
    num_modes_gt: int = typer.Option(
        4,
        "--num-modes-gt",
        "-ng",
        help="Number of ground truth modes.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Device to use for inference and evaluation.",
    ),
):
    """
    Run inference using a pretrained model and immediately evaluate the results.
    This combined command streamlines the workflow by performing both inference
    and evaluation in one step, while keeping predictions and evaluation results
    in separate directories.
    """
    from petimot.infer.infer import infer
    from petimot.eval.eval import evaluate
    import os

    with open(list_path, "r") as f:
        input_list = [line.strip() for line in f if line.strip()]

    print(f"Starting inference phase...")
    infer(
        model_path=model_path,
        config_file=config_file,
        input_list=input_list,
        output_path=prediction_path,
    )

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_prediction_path = os.path.join(prediction_path, model_name)

    print(f"\nStarting evaluation phase...")

    results = evaluate(
        prediction_path=model_prediction_path,
        ground_truth_path=ground_truth_path,
        output_path=evaluation_path,
        sample_ids=input_list,
        num_modes_pred=num_modes_pred,
        num_modes_gt=num_modes_gt,
        device=device,
    )

    print("\nInference and evaluation complete!")
    print(f"Predictions saved to: {model_prediction_path}")
    print(f"Evaluation results saved to: {evaluation_path}")

    return results


if __name__ == "__main__":
    app()
