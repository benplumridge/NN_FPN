import os
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class StaticValidationTests(unittest.TestCase):
    def test_1d_training_samples_total_cross_section(self):
        source = read_source("1D/src/train_model.py")

        self.assertIn(
            "siga = (sigs_max - sigs) * torch.rand(batch_size, device=device)",
            source,
        )
        self.assertIn("sigt = sigs + siga", source)
        self.assertNotIn(
            "sigt = (sigs_max - sigs) * torch.rand(batch_size)",
            source,
        )

    def test_1d_paper_test_case_constants(self):
        source = read_source("1D/src/IC.py")

        self.assertIn("sigs = 100 * x**4", source)
        self.assertIn("z[(x > -0.2) & (x < 0.2)] = 1", source)
        self.assertIn("] = 0.2", source)
        self.assertNotIn("z[(x > -0.2) & (x < 0.2)] = 100", source)
        self.assertNotIn("] = 0.02", source)

    def test_1d_reproduction_config_uses_matching_replica_counts(self):
        for config_path in ["configs/reproduce_1d.yaml", "configs/reproduce_all.yaml"]:
            source = read_source(config_path)
            replicate_match = re.search(r"num_replicates: (\d+)", source)
            tests_match = re.search(r"num_tests: (\d+)", source)
            self.assertIsNotNone(replicate_match)
            self.assertIsNotNone(tests_match)
            self.assertEqual(replicate_match.group(1), tests_match.group(1))
            self.assertIn("device: auto", source)
            self.assertNotIn("device: gpu", source)

    def test_1d_augmented_training_data_flag_is_wired(self):
        train_source = read_source("1D/src/train_model.py")
        self.assertIn('params.get("training_data"', train_source)
        self.assertIn('"augmented": "augmented"', train_source)
        self.assertIn("def _sample_augmented_training_batch", train_source)
        self.assertIn('step_params["tt_flag"] = 1', train_source)
        self.assertIn('params.get("training_time_horizons")', train_source)
        self.assertIn('params.get("aux_moment_loss_weight", 0.0)', train_source)

        for config_path in [
            "configs/reproduce_all.yaml",
            "configs/reproduce_1d.yaml",
            "configs/smoke_1d.yaml",
        ]:
            source = read_source(config_path)
            self.assertRegex(source, r"training_data: (paper|augmented)")
            self.assertRegex(source, r"model_tag: (paper|augmented)")
            self.assertIn("training_time_horizons: [0.25, 0.5, 1.0]", source)
            self.assertIn("aux_moment_loss_weight: 0.0", source)

    def test_1d_model_tags_are_wired_through_training_and_eval(self):
        common_source = read_source("1D/src/params_common.py")
        self.assertIn("def model_tag_from_params", common_source)
        self.assertIn("def tagged_model_path", common_source)
        self.assertIn('"trained_models/model_{tag}_N{N}_{model_idx}.pth"', common_source)
        self.assertIn('"trained_models_const/model_{tag}_N{N}_{model_idx}.pth"', common_source)

        main_source = read_source("main.py")
        self.assertIn('tagged_model_path(N, j, filter_type, params["model_tag"])', main_source)
        self.assertIn('params["model_tag"] = model_tag_from_params(params)', main_source)
        self.assertIn("def default_1d_model_tag", main_source)
        self.assertIn('params_payload["model_tag"] = default_1d_model_tag(dim_cfg)', main_source)

        test_model_source = read_source("1D/src/test_model.py")
        self.assertIn("model_tag = model_tag_from_params(params)", test_model_source)
        self.assertIn("def load_model(N, model_idx, filter_type, model_tag=None):", test_model_source)
        self.assertIn("return tagged_model_path(N, model_idx, filter_type, model_tag)", test_model_source)

        for source_path in [
            "1D/scripts/test_iters.py",
            "1D/scripts/test_iters_reeds.py",
            "1D/scripts/train_all.py",
            "1D/scripts/train_driver.py",
        ]:
            source = read_source(source_path)
            self.assertIn("tagged_model_path", source)
            self.assertIn("model_tag_from_params", source)

        for config_path in [
            "configs/reproduce_all.yaml",
            "configs/reproduce_1d.yaml",
            "configs/smoke_1d.yaml",
        ]:
            source = read_source(config_path)
            self.assertRegex(source, r"model_tag: (paper|augmented)")

    def test_2d_homogeneous_scattering_is_linear(self):
        source = read_source("2D/src/funcs_common.py")

        self.assertIn(
            "scattering = sigs[:, None, None] * psi_prev[:, :, :, 0]",
            source,
        )
        self.assertNotIn(
            "sigs[:, None, None] ** psi_prev[:, :, :, 0]",
            source,
        )

    def test_default_modes_match_nn_reproduction(self):
        self.assertRegex(read_source("1D/src/params_common.py"), r"(?m)^obj_idx = 0$")
        self.assertRegex(
            read_source("2D/src/params_common.py"), r"(?m)^filter_type = 0$"
        )

    def test_log_feature_variants_are_configurable_and_invariant_safe(self):
        for source_path in ["1D/src/funcs_common.py", "2D/src/funcs_common.py"]:
            source = read_source(source_path)
            self.assertIn('"baseline_norm"', source)
            self.assertIn('"log_norm"', source)
            self.assertIn('"baseline_plus_log"', source)
            self.assertIn('"log_material_only"', source)
            self.assertIn('"no_norm_log"', source)
            self.assertIn("def nn_feature_count", source)
            self.assertIn("def _log_magnitude", source)
            self.assertIn("include_material_ratios", source)
            self.assertIn("feature_log_clip", source)
            self.assertIn("feature_normalization", source)
            self.assertIn("material_feature_normalization", source)

        source_2d = read_source("2D/src/funcs_common.py")
        self.assertIn("def _invariant_norm_features", source_2d)
        self.assertIn("psi_norms, dpsi_norms = _invariant_norm_features", source_2d)
        self.assertIn("sigs=sigs", source_2d)
        self.assertIn("sigt=sigt", source_2d)

        source_1d = read_source("1D/src/funcs_common.py")
        self.assertIn("sigs=sigs", source_1d)
        self.assertIn("sigt=sigt", source_1d)

        main_source = read_source("main.py")
        self.assertIn("from funcs_common import nn_feature_count", main_source)
        self.assertIn('params["num_features"] = nn_feature_count(N, params)', main_source)

        for config_path in [
            "configs/reproduce_all.yaml",
            "configs/reproduce_1d.yaml",
            "configs/reproduce_2d.yaml",
            "configs/smoke_1d.yaml",
        ]:
            source = read_source(config_path)
            self.assertRegex(
                source,
                r"feature_variant: (baseline_norm|log_norm|baseline_plus_log|log_material_only|no_norm_log)",
            )
            self.assertIn("feature_normalization: sample", source)
            self.assertIn("material_feature_normalization: none", source)
            self.assertIn("feature_log_clip: [0.0, 20.0]", source)
            self.assertRegex(source, r"include_material_scale_features: (false|true)")
            self.assertRegex(source, r"include_material_ratios: (false|true)")

    def test_training_scripts_set_expected_modes_and_rates(self):
        self.assertIn('params["obj_idx"] = 0', read_source("1D/scripts/train_all.py"))
        self.assertIn(
            'params["obj_idx"] = 0', read_source("1D/scripts/train_driver.py")
        )
        self.assertIn(
            'params["learning_rate"] = 1e-1', read_source("1D/scripts/train_all.py")
        )

        self.assertIn(
            'params["filter_type"] = 0', read_source("2D/scripts/train_all.py")
        )
        self.assertIn(
            'params["filter_type"] = 0', read_source("2D/scripts/train_driver.py")
        )
        self.assertIn(
            'params["learning_rate"] = 1e-2', read_source("2D/scripts/train_all.py")
        )

    def test_reproduction_scripts_have_consistent_imports_and_grid_setup(self):
        self.assertIn(
            "from funcs_common import filter_func",
            read_source("1D/scripts/test_all.py"),
        )
        self.assertIn(
            "from funcs_common import filter_func",
            read_source("1D/scripts/run_script.py"),
        )
        self.assertIn(
            'params["x_edges"] = x_edges', read_source("2D/scripts/test_all.py")
        )

    def test_plot_entry_points_create_expected_outputs(self):
        self.assertIn(
            "def testing(params, model_idx=0):",
            read_source("1D/src/test_model.py"),
        )
        self.assertIn(
            'os.makedirs(f"results/{ic_type}", exist_ok=True)',
            read_source("2D/src/test_model.py"),
        )

    def test_simulation_table_scripts_write_csv_outputs(self):
        for source_path in [
            "1D/scripts/test_iters.py",
            "1D/scripts/test_iters_reeds.py",
            "2D/scripts/test_all.py",
            "2D/scripts/test_all_hohl.py",
        ]:
            source = read_source(source_path)
            self.assertIn("import csv", source)
            self.assertIn("csv.DictWriter", source)
            self.assertIn("writer.writeheader()", source)
            self.assertIn("writer.writerows", source)
            self.assertIn("csv_name", source)

        self.assertIn("mean_flux_error_reduction", read_source("1D/scripts/test_iters.py"))
        self.assertIn("std_flux_error_reduction", read_source("1D/scripts/test_iters_reeds.py"))
        self.assertIn("flux_error_reduction", read_source("2D/scripts/test_all.py"))
        self.assertIn('filter_type = params["filter_type"]', read_source("2D/scripts/test_all_hohl.py"))

    def test_workflow_script_runs_both_ansatz_paths(self):
        script_path = ROOT / "scripts/run_nn_const_workflow.sh"
        source = read_source("scripts/run_nn_const_workflow.sh")

        self.assertTrue(os.access(script_path, os.X_OK))
        self.assertIn("run_1d_workflow", source)
        self.assertIn("run_2d_workflow", source)
        self.assertIn("train_1d 1 nn", source)
        self.assertIn("train_1d 3 const", source)
        self.assertIn("train_2d 0 nn", source)
        self.assertIn("train_2d 1 const", source)
        self.assertIn("results_${label}", source)
        self.assertIn("have_2d_paper_refs", source)

    def test_main_entrypoint_and_yaml_configs_exist(self):
        main_source = read_source("main.py")

        self.assertIn("argparse.ArgumentParser", main_source)
        self.assertIn("--config", main_source)
        self.assertIn("--target", main_source)
        self.assertIn("--ansatz", main_source)
        self.assertIn("--phase", main_source)
        self.assertIn("yaml.safe_load", main_source)
        self.assertIn("TRAIN_1D_CODE", main_source)
        self.assertIn("TRAIN_2D_CODE", main_source)
        self.assertIn("RUN_SCRIPT_CODE", main_source)

        for config_path in [
            "configs/reproduce_all.yaml",
            "configs/reproduce_1d.yaml",
            "configs/reproduce_2d.yaml",
            "configs/smoke_1d.yaml",
        ]:
            source = read_source(config_path)
            self.assertIn("workflow:", source)
            self.assertIn("ansatz:", source)
            self.assertIn("train:", source)
            self.assertIn("simulate:", source)

    def test_requirements_include_yaml_parser(self):
        requirements = read_source("requirements.txt")
        self.assertIn("PyYAML", requirements)

    def test_constant_filter_training_projects_to_nonnegative_values(self):
        train_1d = read_source("1D/src/train_model.py")
        train_2d = read_source("2D/src/train_model.py")

        self.assertIn("if filter_type == 3:", train_1d)
        self.assertIn("NN_model.const.clamp_(min=0.0)", train_1d)
        self.assertIn("if filter_type == 1:", train_2d)
        self.assertIn("NN_model.const.clamp_(min=0.0)", train_2d)

    def test_wandb_logger_is_wired_for_training(self):
        for source_path in ["1D/src/train_model.py", "2D/src/train_model.py"]:
            source = read_source(source_path)
            self.assertIn("from wandb_utils import finish_run, init_wandb, log_metrics", source)
            self.assertIn("wandb_run = init_wandb(params)", source)
            self.assertIn("log_metrics(", source)
            self.assertIn("finish_run(wandb_run)", source)
            self.assertIn('"train/loss"', source)
            self.assertIn('"train/filter_strength_mean"', source)

        for source_path in ["1D/src/wandb_utils.py", "2D/src/wandb_utils.py"]:
            source = read_source(source_path)
            self.assertIn("def init_wandb(params):", source)
            self.assertIn("wandb.init(", source)
            self.assertIn("mode = cfg.get", source)
            self.assertIn("or \"online\"", source)
            self.assertIn("def log_metrics", source)

    def test_training_hot_paths_avoid_gpu_synchronization_regressions(self):
        train_1d = read_source("1D/src/train_model.py")
        train_2d = read_source("2D/src/train_model.py")
        common_1d = read_source("1D/src/funcs_common.py")
        common_2d = read_source("2D/src/funcs_common.py")

        for source in [train_1d, train_2d]:
            self.assertNotIn('.to("cpu")', source)
            self.assertIn("torch.rand(batch_size, device=device)", source)
            self.assertIn("should_log_metrics(params, l, final=final_epoch)", source)

        self.assertIn("_TIMESTEPPING_TENSOR_CACHE", common_1d)
        self.assertIn("torch.arange(1, N + 1, dtype=torch.float32, device=device)", common_1d)
        self.assertIn("torch.linalg.eigh(A)", common_1d)
        self.assertIn("filter_coeffs,", common_1d)
        self.assertNotIn("torch.linalg.eig(", common_1d)

        self.assertIn("def compute_upwind_matrices(N, device):", common_2d)
        self.assertIn("_SOLVER_TENSOR_CACHE", common_2d)
        self.assertIn("filter_coeffs = filter_coefficients(filter_order, N, num_basis, device=device)", common_2d)
        self.assertIn("upwind_matrices = compute_upwind_matrices(N, device=device)", common_2d)
        self.assertNotIn("filter_coeffs, upwind_matrices = solver_tensors", common_2d)
        self.assertIn("def upwind_flux(N, num_basis, psi, params, upwind_matrices):", common_2d)
        self.assertIn("torch.linalg.eigh(Ax)", common_2d)
        self.assertNotIn("torch.linalg.eig(", common_2d)

    def test_wandb_config_and_dependency_are_present(self):
        self.assertIn("wandb", read_source("requirements.txt"))
        for config_path in [
            "configs/reproduce_all.yaml",
            "configs/reproduce_1d.yaml",
            "configs/reproduce_2d.yaml",
            "configs/smoke_1d.yaml",
        ]:
            source = read_source(config_path)
            self.assertIn("wandb:", source)
            self.assertIn("project: NN_FPN", source)
            self.assertRegex(source, r"mode: (offline|online)")
            self.assertNotIn("device: cpu", source)
            self.assertNotIn("device: gpu", source)
            self.assertNotIn("device: cuda", source)

        reproduce_1d = read_source("configs/reproduce_1d.yaml")
        self.assertRegex(reproduce_1d, r"enabled: (false|true)")
        self.assertRegex(reproduce_1d, r"mode: (offline|online)")
        self.assertIn("device: auto", reproduce_1d)

        main_source = read_source("main.py")
        self.assertIn('"wandb": merged_dict(', main_source)
        self.assertIn('workflow.get("wandb", {})', main_source)
        self.assertIn('params["wandb"] = wandb_cfg', main_source)
        self.assertIn('resolve_device(params.get("device"))', main_source)


if __name__ == "__main__":
    unittest.main()
