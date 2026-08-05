import autograd.numpy as npa
import numpy as np
import pytest
from autograd import grad

from msopt import Sub_Mapping
from msopt.Opt_MS2 import Mapping


def test_fixed_material_masks_are_exact_when_disjoint():
    density = npa.array([0.2, 0.4, 0.6, 0.8])
    one_mask = np.array([True, False, False, False])
    zero_mask = np.array([False, True, True, False])

    result = Sub_Mapping.apply_fixed_material_masks(
        density,
        one_mask=one_mask,
        zero_mask=zero_mask,
    )

    np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.8])


def test_fixed_material_masks_reject_overlap():
    density = npa.array([0.2, 0.4, 0.6, 0.8])
    one_mask = np.array([True, True, False, False])
    zero_mask = np.array([False, True, True, False])

    with pytest.raises(ValueError, match="must be disjoint.*1 overlapping"):
        Sub_Mapping.apply_fixed_material_masks(
            density,
            one_mask=one_mask,
            zero_mask=zero_mask,
        )


def test_fixed_material_masks_zero_the_autograd_derivative():
    one_mask = np.array([True, False, False, False])
    zero_mask = np.array([False, True, False, False])

    def summed_density(values):
        return npa.sum(
            Sub_Mapping.apply_fixed_material_masks(
                values,
                one_mask=one_mask,
                zero_mask=zero_mask,
            )
        )

    derivative = grad(summed_density)(npa.array([0.2, 0.4, 0.6, 0.8]))
    np.testing.assert_allclose(derivative, [0.0, 0.0, 1.0, 1.0])


def test_rotated_rectangle_mask_matches_lumerical_local_coordinates():
    axis = np.linspace(-1.0, 1.0, 5)
    mask = Sub_Mapping.rotated_rectangle_mask_2d(
        axis,
        axis,
        span_1=0.5,
        span_2=3.0,
        rotation_deg=-45.0,
    )

    # A thin local-axis-1 span rotated -45 degrees follows axis_1=axis_2.
    assert np.all(np.diag(mask))
    assert not mask[0, -1]
    assert not mask[-1, 0]


def test_mapping_is_waveguide_branch_applies_custom_masks_after_projection():
    ny, nz, nx = 7, 7, 2
    one_mask = np.zeros((ny, nz), dtype=bool)
    zero_mask = np.zeros((ny, nz), dtype=bool)
    one_mask[:3, 3] = True
    zero_mask[4, :] = True

    mapping = Mapping(
        Symmetry_sim=False,
        Sym_geo_width=False,
        Sym_geo_length=False,
        Is_waveguide=[True, False, False, 2],
        DR_info=[0.1, 0.6, 0.6, 1, 2, 0],
        DR_N_info=[nx, ny, nz, 10],
        Mask_info=[0.0, 0.0],
        Mask_pixels=1,
        MFS=0.1,
        MGS=0.1,
        Fixed_waveguide_masks={
            "one": one_mask,
            "zero": zero_mask,
        },
    )
    result = np.asarray(mapping(npa.full(ny * nz, 0.4), beta=2.0))
    result = result.reshape(nx, ny, nz)

    assert not np.any(one_mask & zero_mask)
    assert np.all(result[:, one_mask] == 1.0)
    assert np.all(result[:, zero_mask] == 0.0)
