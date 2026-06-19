import json
import os

def main():
    files_to_write = [
        "plot_analytical_2d.py",
        "geometry.py",
        "network.py",
        "evaluator.py",
        "main.py",
        "plot_all_fields_2d.py"
    ]

    cells = []

    # Intro cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Kirsch 2D PINN - Google Colab Training Notebook (Full Domain)\n",
            "本 Notebook 会将相关的 Python 源码直接写入 Colab 的文件系统，然后启动训练。\n",
            "运行前请确保在 Colab 中启用了 GPU (Runtime -> Change runtime type -> T4 GPU)."
        ]
    })

    for fname in files_to_write:
        if not os.path.exists(fname):
            continue
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by lines and add newlines back, except for the last line
        lines = content.split('\n')
        source_lines = [f"%%writefile {fname}\n"]
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                source_lines.append(line + '\n')
            else:
                source_lines.append(line)
                
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        })

    # Run main cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python main.py"
        ]
    })

    # Display result
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "display(Image('result_2d_kirsch_comparison.png'))\n",
            "display(Image('result_2d_kirsch.png'))"
        ]
    })

    # Run plot all fields
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python plot_all_fields_2d.py"
        ]
    })

    # Display all fields
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image('result_2d_kirsch_all_fields.png'))"
        ]
    })

    # Launch Tensorboard
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%load_ext tensorboard\n",
            "%tensorboard --logdir runs/"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("Kirsch_2D_Colab.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
