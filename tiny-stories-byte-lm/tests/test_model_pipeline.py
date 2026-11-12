import random
import tempfile
import unittest
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.data import MappedByteStories, write_byte_dataset
from tinystories_byte_lm.generation import generate_bytes, is_valid_utf8
from tinystories_byte_lm.model import ByteTransformer, TeLU, trainable_parameter_count
from tinystories_byte_lm.training import (
    TrainConfig,
    load_checkpoint,
    run_training,
    save_checkpoint,
    train_step,
)


def tiny_config(position_encoding: str = "nope") -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        context_length=8,
        position_encoding=position_encoding,  # type: ignore[arg-type]
        lut_seed=19,
    )


class ModelPipelineTests(unittest.TestCase):
    def test_fixed_lut_has_no_gradient_and_output_is_not_tied(self) -> None:
        model = ByteTransformer(tiny_config())

        self.assertFalse(model.byte_lut.weight.requires_grad)
        self.assertIsNot(model.byte_lut.weight, model.output_projection.weight)
        self.assertGreater(trainable_parameter_count(model), 0)
        self.assertFalse(
            any(
                parameter is model.byte_lut.weight
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_future_bytes_do_not_change_earlier_causal_logits(self) -> None:
        torch.manual_seed(10)
        model = ByteTransformer(tiny_config()).eval()
        first = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        changed_future = torch.tensor([[10, 11, 200, 201]], dtype=torch.long)

        with torch.no_grad():
            first_logits = model(first)
            changed_logits = model(changed_future)

        torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])

    def test_rope_configuration_runs_forward(self) -> None:
        model = ByteTransformer(tiny_config("rope"))
        logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))

        self.assertEqual(logits.shape, (1, 3, 256))

    def test_telu_is_finite_for_extreme_values_in_supported_dtypes(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                values = torch.tensor([-100.0, -5.0, 0.0, 5.0, 100.0], dtype=dtype)
                values.requires_grad_()
                output = TeLU()(values)
                output.float().sum().backward()

                self.assertEqual(output.dtype, dtype)
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertIsNotNone(values.grad)
                self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_training_step_and_checkpoint_restore(self) -> None:
        torch.manual_seed(10)
        config = tiny_config()
        model = ByteTransformer(config)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        inputs = torch.tensor([[65, 66, 0], [120, 121, 122]], dtype=torch.long)
        targets = torch.tensor([[66, 67, -100], [121, 122, 123]], dtype=torch.long)
        metrics = train_step(model, optimizer, inputs, targets)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
        self.assertEqual(metrics.input_bytes, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "resume.pt"
            save_checkpoint(checkpoint, model, optimizer, step=7, config={"seed": 10})
            restored_model = ByteTransformer(config)
            restored_optimizer = torch.optim.AdamW(
                (
                    parameter
                    for parameter in restored_model.parameters()
                    if parameter.requires_grad
                ),
                lr=1e-3,
            )
            step, saved_config = load_checkpoint(
                checkpoint, restored_model, restored_optimizer
            )

        self.assertEqual(step, 7)
        self.assertEqual(saved_config, {"seed": 10})
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), restored_model(inputs))

    def test_mapped_byte_windows_never_cross_story_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "stories.bin"
            index_path = Path(temp_dir) / "stories.idx"
            result = write_byte_dataset([b"abc", b"XY"], data_path, index_path)

            with MappedByteStories(data_path, index_path) as dataset:
                first_story_inputs, first_story_targets = dataset.window_at(1, 8)
                second_story_inputs, second_story_targets = dataset.window_at(2, 8)
                inputs, targets = dataset.sample_batch(4, 4, random.Random(7))

        self.assertEqual(result["next_byte_input_bytes"], 3)
        self.assertEqual(first_story_inputs.tolist(), [ord("b")])
        self.assertEqual(first_story_targets.tolist(), [ord("c")])
        self.assertEqual(second_story_inputs.tolist(), [ord("X")])
        self.assertEqual(second_story_targets.tolist(), [ord("Y")])
        self.assertEqual(inputs.shape, targets.shape)
        valid_targets = int((targets != -100).sum().item())
        self.assertGreaterEqual(valid_targets, 4)
        self.assertLessEqual(valid_targets, 16)

    def test_training_run_resumes_deterministically_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_data = root / "train.bin"
            train_index = root / "train.idx"
            validation_data = root / "validation.bin"
            validation_index = root / "validation.idx"
            write_byte_dataset(
                [b"abcdefghijk", b"mnopqrstuvw"],
                train_data,
                train_index,
            )
            write_byte_dataset(
                [b"abcdefghijk", b"mnopqrstuvw"],
                validation_data,
                validation_index,
            )
            config = tiny_config()
            train_config = TrainConfig(
                steps=2,
                batch_size=2,
                learning_rate=1e-3,
                eval_interval=1,
                eval_batches=1,
                checkpoint_interval=1,
                seed=29,
                num_threads=1,
            )
            uninterrupted_dir = root / "uninterrupted"
            resumed_dir = root / "resumed"

            uninterrupted = run_training(
                config,
                train_config,
                train_data,
                train_index,
                validation_data,
                validation_index,
                uninterrupted_dir,
            )
            resumed = run_training(
                config,
                train_config,
                train_data,
                train_index,
                validation_data,
                validation_index,
                resumed_dir,
                uninterrupted_dir / "checkpoint-00000001.pt",
            )
            uninterrupted_checkpoint = torch.load(
                uninterrupted_dir / "checkpoint-00000002.pt",
                map_location="cpu",
                weights_only=True,
            )
            resumed_checkpoint = torch.load(
                resumed_dir / "checkpoint-00000002.pt",
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual([entry["step"] for entry in uninterrupted], [1, 2])
        self.assertEqual([entry["step"] for entry in resumed], [2])
        for key, tensor in uninterrupted_checkpoint["model_state"].items():
            torch.testing.assert_close(tensor, resumed_checkpoint["model_state"][key])
        self.assertIn("validation_byte_perplexity", resumed[0])
        self.assertGreater(resumed[0]["peak_process_memory_bytes"], 0)

    def test_generation_requires_a_prompt_and_validity_is_strict(self) -> None:
        model = ByteTransformer(tiny_config())
        generated = generate_bytes(model, b"Once ", max_new_bytes=3, seed=2)

        self.assertEqual(len(generated), len(b"Once ") + 3)
        self.assertTrue(is_valid_utf8("story \u00e9".encode("utf-8")))
        self.assertFalse(is_valid_utf8(b"\xc3"))
        with self.assertRaises(ValueError):
            generate_bytes(model, b"", max_new_bytes=1)


if __name__ == "__main__":
    unittest.main()
