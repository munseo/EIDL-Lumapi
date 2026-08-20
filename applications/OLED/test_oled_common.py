import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import msopt as ms
import oled_common as oc
import msopt.Sub_Mapping as sm


class OledPerformanceMetricTests(unittest.TestCase):
    def setUp(self):
        self.spec = oc.make_ratio_performance_spec(
            [0.0, 30.0, 45.0, 60.0],
            [1.0, 0.90, 0.85, 0.0],
            0.05,
        )

    def test_large_normal_power_cannot_hide_bad_angular_ratios(self):
        bad = oc.oled_performance_metrics(
            [10.347089, 0.278530, 0.415692, 0.577394],
            self.spec,
            power_floor=1e-12,
            violation_scale=0.05,
        )
        good = oc.oled_performance_metrics(
            [1.0, 0.90, 0.85, 1e-12],
            self.spec,
            power_floor=1e-12,
            violation_scale=0.05,
        )
        self.assertGreater(good["score"], 100.0 * bad["score"])
        self.assertTrue(good["all_ratio_windows_met"])
        self.assertFalse(bad["all_ratio_windows_met"])

    def test_autograd_matches_finite_difference(self):
        try:
            from autograd import grad
            from autograd import numpy as anp
        except ModuleNotFoundError:
            self.skipTest("autograd is not installed")

        def objective(x):
            return oc.oled_constrained_score(
                x,
                self.spec["ratio_min"],
                self.spec["ratio_max"],
                power_floor=1e-12,
                violation_scale=0.05,
            )

        x = anp.array([1.2, 0.7, 0.6, 0.1])
        analytic = np.asarray(grad(objective)(x))
        eps = 1e-6
        numeric = np.asarray([
            (
                objective(np.asarray(x) + eps * np.eye(x.size)[idx])
                - objective(np.asarray(x) - eps * np.eye(x.size)[idx])
            )
            / (2.0 * eps)
            for idx in range(x.size)
        ])
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)

    def test_direction_cosine_jacobian_matches_upper_hemisphere_integral(self):
        u = np.linspace(-1.0, 1.0, 401)
        _theta, power, _ux, _uy = oc.direction_cosine_power_spectrum(
            np.ones((u.size, u.size)),
            u,
            u,
        )
        # Unit radial intensity integrated over an upper hemisphere is 2*pi.
        self.assertAlmostEqual(float(np.sum(power)) / (2.0 * np.pi), 1.0, delta=0.02)


class MappingDimensionTests(unittest.TestCase):
    def test_freeform_mapping_sets_grid_dimensions(self):
        mapping = ms.Opt_MS2.Mapping(
            DR_info=[1.0, 1.0, 0.2, 0, 1, 2],
            DR_N_info=[10, 12, 6, 5],
            Mask_info=[0.0, 0.0],
            Is_waveguide=[False, False, False, 2],
            Is_freeform=[True, False, False],
            Is_radial_3d=False,
        )
        self.assertEqual(mapping.DR_width, 1.0)
        self.assertEqual(mapping.N_width, 10)
        self.assertEqual(mapping.DR_length, 1.0)
        self.assertEqual(mapping.N_length, 12)
        self.assertEqual(mapping.N_height, 6)


class LayerwiseSymmetryTests(unittest.TestCase):
    def test_c8_symmetry_is_applied_per_layer(self):
        layer0 = np.arange(16, dtype=float).reshape(4, 4)
        layer1 = np.arange(16, dtype=float).reshape(4, 4) + 100.0
        x = np.stack([layer0, layer1], axis=0)
        sym = sm.apply_c8_symmetry_per_layer(x)
        self.assertEqual(sym.shape, x.shape)
        np.testing.assert_allclose(sym[0], sm.apply_c8_symmetry_2d(layer0))
        np.testing.assert_allclose(sym[1], sm.apply_c8_symmetry_2d(layer1))


class OledPostprocessCompletenessTests(unittest.TestCase):
    def test_validated_source_grid_matches_six_by_six_endpoint_protocol(self):
        with tempfile.TemporaryDirectory() as run_dir:
            env = {
                "EIDL_RUN_DIR": run_dir,
                "MSOPT_OLED_PP_DIPOLE_GRID": "6",
                "MSOPT_OLED_PP_SOURCE_LAYOUT": "validated_endpoint",
                "MSOPT_OLED_PP_RESOLUTION": "40",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                G = oc.build_config(period_x_default=1.1)
                points = oc.central_cell_dipoles(G, 20, "x")
        self.assertEqual(len(points), 36)
        coordinates = sorted({point[0] for point in points})
        self.assertEqual(len(coordinates), 6)
        self.assertAlmostEqual(coordinates[0], -(0.55 - 2.0 / 40.0))
        self.assertAlmostEqual(coordinates[-1], +(0.55 - 2.0 / 40.0))

    def test_lumerical_flux_box_strictly_encloses_endpoint_sources(self):
        with tempfile.TemporaryDirectory() as run_dir:
            env = {
                "EIDL_RUN_DIR": run_dir,
                "MSOPT_OLED_PP_DIPOLE_GRID": "6",
                "MSOPT_OLED_PP_SOURCE_LAYOUT": "validated_endpoint",
                "MSOPT_OLED_PP_RESOLUTION": "50",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                G = oc.build_config(period_x_default=2.5)
                points = oc.central_cell_dipoles(G, 20, "x")

        box_inset = 1.0 / G.pp_resolution
        box_size = [
            G.active_x - 2.0 * box_inset,
            G.active_y - 2.0 * box_inset,
            G.eml_h - 2.0 * box_inset,
        ]
        clearance = oc.minimum_source_box_clearance(
            points,
            [0.0, 0.0, G.eml_c[2]],
            box_size,
        )
        self.assertAlmostEqual(clearance, 1.0 / G.pp_resolution)

        overlapping_box_size = [
            G.active_x - 4.0 / G.pp_resolution,
            G.active_y - 4.0 / G.pp_resolution,
            G.eml_h - 4.0 / G.pp_resolution,
        ]
        self.assertAlmostEqual(
            oc.minimum_source_box_clearance(
                points,
                [0.0, 0.0, G.eml_c[2]],
                overlapping_box_size,
            ),
            0.0,
        )

    def test_flux_box_uses_outward_face_signs(self):
        class FakeFdtd:
            values = {
                "xp": 0.30,
                "xm": -0.10,
                "yp": 0.20,
                "ym": -0.05,
                "zp": 0.40,
                "zm": -0.15,
            }

            def transmission(self, name):
                return self.values[name]

        faces = [
            ("xp", +1.0), ("xm", -1.0),
            ("yp", +1.0), ("ym", -1.0),
            ("zp", +1.0), ("zm", -1.0),
        ]
        power = oc.read_flux_box_power(FakeFdtd(), faces, source_power=2.0)
        self.assertAlmostEqual(power, 2.0 * (0.30 + 0.10 + 0.20 + 0.05 + 0.40 + 0.15))

    def test_complete_postprocess_uses_source_box_normalization(self):
        class FakeFdtd:
            def switchtolayout(self):
                return None

            def setnamed(self, *_args):
                return None

            def close(self):
                return None

        class FakeSim:
            def __init__(self):
                self.fdtd = FakeFdtd()

            def add_design_grid(self, *_args):
                return None

            def add_monitor(self, *_args):
                return None

            def run(self, **_kwargs):
                return None

        theta = np.asarray([[0.0, 30.0], [30.0, 45.0]])
        ukx = np.asarray([[0.0, 0.5], [0.0, 0.5]])
        uky = np.asarray([[0.0, 0.0], [0.5, 0.5]])
        raw_spectrum = np.ones((2, 2))

        def transmission(_fdtd, name):
            if name == "FoM_monitor":
                return 0.25
            return {
                "pp_source_box_xp": 0.10,
                "pp_source_box_xm": -0.10,
                "pp_source_box_yp": 0.10,
                "pp_source_box_ym": -0.10,
                "pp_source_box_zp": 0.10,
                "pp_source_box_zm": -0.10,
            }[name]

        with tempfile.TemporaryDirectory() as run_dir:
            env = {
                "EIDL_RUN_DIR": run_dir,
                "MSOPT_OLED_PP_MODE": "single",
                "MSOPT_OLED_PP_DIPOLE_GRID": "1",
                "MSOPT_OLED_POSTPROCESS_POLARIZATIONS": "x",
                "MSOPT_OLED_PP_RETRIES": "0",
                "MSOPT_OLED_PP_FIELD_IMAGES": "0",
                "MSOPT_OLED_PP_RADIATION_IMAGES": "0",
                "MSOPT_OLED_PP_REQUIRE_COMPLETE": "1",
                "MSOPT_OLED_PP_REQUIRE_METRICS": "1",
                "MSOPT_OLED_POSTPROCESS_ANGLE_RES": "19",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                G = oc.build_config(period_x_default=1.1)
                design = np.full(G.design_cells, 0.5)
                with (
                    mock.patch.object(oc, "make_sim", return_value=FakeSim()),
                    mock.patch.object(oc, "add_stack"),
                    mock.patch.object(oc, "delete_object"),
                    mock.patch.object(oc, "add_dipole"),
                    mock.patch.object(oc, "load_run_results"),
                    mock.patch.object(oc, "read_transmission", side_effect=transmission),
                    mock.patch.object(oc, "source_freqs", return_value=np.asarray([1.0])),
                    mock.patch.object(oc, "read_source_power", return_value=2.0),
                    mock.patch.object(oc, "read_dipole_power", return_value=1.0),
                    mock.patch.object(
                        oc,
                        "n2f_spectrum",
                        return_value=(theta, raw_spectrum, ukx, uky),
                    ),
                    mock.patch.object(oc, "save_radiation_map_figure"),
                    mock.patch.object(oc, "render_emission_figure"),
                    mock.patch.object(oc, "save_per_dipole_emission_plot", return_value=None),
                ):
                    manifest = oc.run_postprocess(
                        G,
                        design,
                        performance_spec=self._performance_spec(),
                    )

            # Signed box transmission is 0.6; source power is 2.0, so the
            # validated normalization is 1.2. Top power is 0.25*2.0 = 0.5.
            self.assertAlmostEqual(manifest["ensemble_lee"], 0.5 / 1.2)
            self.assertEqual(manifest["power_normalization"]["reference"], "source_flux_box")
            self.assertEqual(manifest["status"], "complete")

    def test_incomplete_dipole_sweep_is_not_published(self):
        class FakeFdtd:
            def switchtolayout(self):
                return None

            def setnamed(self, *_args):
                return None

            def close(self):
                return None

        class FakeSim:
            def __init__(self):
                self.fdtd = FakeFdtd()

            def add_design_grid(self, *_args):
                return None

            def add_monitor(self, *_args):
                return None

            def run(self, **_kwargs):
                return None

        with tempfile.TemporaryDirectory() as run_dir:
            env = {
                "EIDL_RUN_DIR": run_dir,
                "MSOPT_OLED_PP_MODE": "single",
                "MSOPT_OLED_PP_DIPOLE_GRID": "1",
                "MSOPT_OLED_POSTPROCESS_POLARIZATIONS": "x",
                "MSOPT_OLED_PP_RETRIES": "0",
                "MSOPT_OLED_PP_FIELD_IMAGES": "0",
                "MSOPT_OLED_PP_RADIATION_IMAGES": "0",
                "MSOPT_OLED_PP_REQUIRE_COMPLETE": "1",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                G = oc.build_config(period_x_default=1.1)
                design = np.full(G.design_cells, 0.5)
                with (
                    mock.patch.object(oc, "make_sim", return_value=FakeSim()),
                    mock.patch.object(oc, "add_stack"),
                    mock.patch.object(oc, "delete_object"),
                    mock.patch.object(oc, "add_dipole"),
                    mock.patch.object(oc, "load_run_results", side_effect=RuntimeError("missing monitor")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "incomplete"):
                        oc.run_postprocess(G, design, performance_spec=self._performance_spec())

                manifest_path = os.path.join(G.design_dir, "OLED_postprocess_manifest.json")
                with open(manifest_path, encoding="utf-8") as fp:
                    manifest = json.load(fp)
                self.assertEqual(manifest["status"], "incomplete")
                self.assertFalse(manifest["authoritative"])
                self.assertEqual(manifest["requested_runs"], 1)
                self.assertEqual(manifest["successful_runs"], 0)

    @staticmethod
    def _performance_spec():
        return oc.make_ratio_performance_spec(
            [0.0, 30.0, 45.0, 60.0],
            [1.0, 0.90, 0.85, 0.0],
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
