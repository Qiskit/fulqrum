# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from dataclasses import dataclass, field


def address(array) -> int:

    return 0 if array is None else int(array.data.ptr)


def offsets(array) -> int:

    if array is None:
        return 0
    if str(array.dtype) != "int64":
        raise TypeError(
            f"edge offset tables must be int64 (the kernels read int64), got {array.dtype}"
        )
    return int(array.data.ptr)


@dataclass(frozen=True)
class CliqueCross:


    residual_off: object
    residual_lengths: object
    residual_pack: object
    residual_order: object
    beta_words: object
    tasks: object
    residual_edges: int
    edge_words: object
    group_offsets: object
    group_pairs: object
    tile_group_offsets: object
    parent_offsets: object
    members: object
    bucket_ids: object
    coefficients: object
    bucket_start: tuple
    num_tiles: int
    tile_rows: int
    edges_per_batch: int
    orbitals: int
    transposed: bool
    parent_group: int = 0
    mma: bool = False
    distinct_rows: bool = False
    panel: object = None
    nonfinite: object = None
    panel_columns: int = 0
    panel_batch: tuple = ()
    lbg: object = None
    lbg_scratch: object = None
    lbg_flag: object = None


@dataclass(frozen=True)
class BetaDoubleClass:


    blocks: object
    block_offsets: object
    tile_info: object
    row_members: object
    col_members: object
    nonfinite: object
    num_tiles: int
    alpha_group: int
    residual_offsets: object
    residual_columns: object
    residual_values: object


@dataclass(frozen=True)
class AlphaDoubleClass:


    classes: object
    xt: object
    yt: object
    slab: int
    n_alpha_src: int
    residual: object
    leftover: object = None
    slab_identity: object = None


@dataclass
class SourceLaunch:


    geometry: tuple
    src_in: int
    out: int
    alpha_slot: int
    beta_slot: int
    stream: int
    same_alpha: bool
    same_beta: bool
    view: dict
    caches: dict
    pair: object
    rows: dict
    num_alpha: int
    num_beta: int
    alpha_strings: object
    beta_strings: object
    coefficients: object
    cross: object = None
    order: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    arm_events: object = None
    bbbb_class: object = None
    aaaa_class: object = None


def _check(launch_call, arm: str, launch=None) -> None:


    if launch is None or launch.arm_events is None:
        try:
            launch_call()
        except RuntimeError as error:
            raise RuntimeError(f"{arm} arm: {error}") from error
        return
    import cupy  # noqa: PLC0415

    stream = cupy.cuda.ExternalStream(launch.stream)
    start, end = cupy.cuda.Event(), cupy.cuda.Event()
    start.record(stream)
    try:
        launch_call()
    except RuntimeError as error:
        raise RuntimeError(f"{arm} arm: {error}") from error
    end.record(stream)
    launch.arm_events.append((arm, start, end))


def _record(launch: SourceLaunch, name: str) -> None:

    event = launch.events.get(name)
    if event is not None:
        import cupy  # noqa: PLC0415

        event.record(cupy.cuda.ExternalStream(launch.stream))


def _alpha_doubles(k, launch: SourceLaunch) -> None:

    view, cache = launch.view, launch.caches.get("aaaa")
    if not (
        launch.same_beta
        and cache is not None
        and view.get("d_alpha4_sell_off") is not None
        and view.get("d_alpha4_sell_len") is not None
        and view.get("d_alpha4_sell_j") is not None
    ):
        return
    pair = launch.pair
    form = launch.aaaa_class
    if form is not None:
        c = form.classes
        _check(
            lambda: k.launch_alpha_doubles_class(
                launch.geometry,
                launch.src_in,
                launch.out,
                launch.alpha_slot,
                launch.beta_slot,
                address(c.blocks),
                offsets(c.block_offsets),
                address(c.tile_info),
                address(c.row_members),
                address(c.col_members),
                address(c.nonfinite),
                c.num_tiles,
                c.alpha_group,
                address(form.xt),
                address(form.yt),
                form.slab,
                form.n_alpha_src,
                launch.stream,
                *(
                    (
                        offsets(form.leftover[0]),
                        address(form.leftover[1]),
                        address(form.leftover[2]),
                        address(form.slab_identity),
                    )
                    if form.leftover is not None
                    else (0, 0, 0, 0)
                ),
                launch.rows["bbbb"],
            ),
            "alpha doubles class",
            launch,
        )
        if form.residual is None:
            return
        sell = form.residual
    else:
        sell = (
            view["d_alpha4_sell_off"],
            view["d_alpha4_sell_len"],
            view["d_alpha4_sell_j"],
            cache,
        )
    _check(
        lambda: k.launch_alpha_doubles(
            launch.geometry,
            launch.src_in,
            launch.out,
            launch.alpha_slot,
            launch.beta_slot,
            offsets(sell[0]),
            address(sell[1]),
            address(sell[2]),
            address(sell[3]),
            launch.rows["aaaa"],
            pair.aaaa_min_blocks if pair is not None else 0,
            bool(pair is not None and pair.aaaa_identity_slot),
            launch.stream,
            int(launch.order.get("aaaa", 0)),
        ),
        "alpha doubles",
        launch,
    )


def _alpha_singles(k, launch: SourceLaunch) -> None:

    view, caches, pair = launch.view, launch.caches, launch.pair
    pair_form = pair is not None and bool(pair.aa_enable)
    if not (
        launch.same_beta
        and view.get("d_alpha2_ptr") is not None
        and view.get("d_alpha2_j") is not None
        and launch.beta_strings is not None
        and (
            pair_form
            or (caches.get("aa_c0") is not None and caches.get("aa_cpb") is not None)
        )
    ):
        return
    if pair_form:
        _check(
            lambda: k.launch_alpha_singles_pair(
                launch.geometry,
                launch.src_in,
                launch.out,
                launch.alpha_slot,
                launch.beta_slot,
                int(pair.aa_sell_off),
                int(pair.aa_sell_len),
                int(pair.aa_sell_pack),
                int(pair.aa_c0_sell),
                int(pair.aa_t),
                int(pair.aa_pairs),
                launch.rows["aa"],
                launch.stream,
                bool(launch.order.get("aa_hoist", False)),
            ),
            "alpha singles",
            launch,
        )
        return
    _check(
        lambda: k.launch_alpha_singles_delta(
            launch.geometry,
            launch.src_in,
            launch.out,
            launch.alpha_slot,
            launch.beta_slot,
            offsets(view["d_alpha2_ptr"]),
            address(view["d_alpha2_j"]),
            address(caches["aa_c0"]),
            address(caches["aa_cpb"]),
            address(launch.beta_strings),
            launch.num_beta,
            launch.rows["aa"],
            launch.stream,
        ),
        "alpha singles",
        launch,
    )


def _beta_doubles(k, launch: SourceLaunch) -> None:

    view, cache = launch.view, launch.caches.get("bbbb")
    if not (
        launch.same_alpha
        and cache is not None
        and view.get("d_beta4_ptr") is not None
        and view.get("d_beta4_j") is not None
    ):
        return
    form = launch.bbbb_class
    if form is not None:
        if form.num_tiles:
            _check(
                lambda: k.launch_beta_doubles_class(
                    launch.geometry,
                    launch.src_in,
                    launch.out,
                    launch.alpha_slot,
                    launch.beta_slot,
                    address(form.blocks),
                    offsets(form.block_offsets),
                    address(form.tile_info),
                    address(form.row_members),
                    address(form.col_members),
                    address(form.nonfinite),
                    form.num_tiles,
                    form.alpha_group,
                    launch.stream,
                ),
                "beta doubles class",
                launch,
            )
        if form.residual_values is not None:
            _check(
                lambda: k.launch_beta_doubles(
                    launch.geometry,
                    launch.src_in,
                    launch.out,
                    launch.alpha_slot,
                    launch.beta_slot,
                    offsets(form.residual_offsets),
                    address(form.residual_columns),
                    address(form.residual_values),
                    launch.rows["bbbb"],
                    launch.stream,
                    int(launch.order.get("bbbb", 0)),
                ),
                "beta doubles",
                launch,
            )
        return
    _check(
        lambda: k.launch_beta_doubles(
            launch.geometry,
            launch.src_in,
            launch.out,
            launch.alpha_slot,
            launch.beta_slot,
            offsets(view["d_beta4_ptr"]),
            address(view["d_beta4_j"]),
            address(cache),
            launch.rows["bbbb"],
            launch.stream,
            int(launch.order.get("bbbb", 0)),
        ),
        "beta doubles",
        launch,
    )


def _beta_singles(k, launch: SourceLaunch) -> None:

    view, caches, pair = launch.view, launch.caches, launch.pair
    pair_form = pair is not None and bool(pair.bb_enable)
    if not (
        launch.same_alpha
        and view.get("d_beta2_ptr") is not None
        and view.get("d_beta2_j") is not None
        and caches.get("bb_c0") is not None
        and launch.alpha_strings is not None
        and (pair_form or caches.get("bb_cpb") is not None)
    ):
        return
    if pair_form:
        _check(
            lambda: k.launch_beta_singles_pair(
                launch.geometry,
                launch.src_in,
                launch.out,
                launch.alpha_slot,
                launch.beta_slot,
                offsets(view["d_beta2_ptr"]),
                address(view["d_beta2_j"]),
                address(caches["bb_c0"]),
                int(pair.bb_word),
                int(pair.bb_t),
                int(pair.bb_t_cols),
                launch.rows.get("bb_pair", launch.rows["bb"]),
                launch.stream,
                int(launch.order.get("bb_pair", 0)),
                bool(launch.order.get("bb_hoist", False)),
            ),
            "beta singles",
            launch,
        )
        return
    _check(
        lambda: k.launch_beta_singles_delta(
            launch.geometry,
            launch.src_in,
            launch.out,
            launch.alpha_slot,
            launch.beta_slot,
            offsets(view["d_beta2_ptr"]),
            address(view["d_beta2_j"]),
            address(caches["bb_c0"]),
            address(caches["bb_cpb"]),
            address(launch.alpha_strings),
            launch.num_alpha,
            launch.rows["bb"],
            launch.stream,
        ),
        "beta singles",
        launch,
    )


def _cross_ready(launch: SourceLaunch) -> bool:

    view = launch.view
    return (
        launch.cross is not None
        and launch.coefficients is not None
        and all(
            view.get(name) is not None
            for name in (
                "d_alpha2_sell_off",
                "d_alpha2_sell_len",
                "d_alpha2_sell_pack",
                "d_beta2_ptr",
                "d_beta2_j",
                "d_beta2_p0",
                "d_beta2_p1",
            )
        )
    )


def _clique_tuple(
    clique: CliqueCross,
    bucket: int,
    first: int,
    count: int,
    panel: int,
    flag: int,
    columns: int,
):

    return (
        address(clique.edge_words),
        address(clique.group_offsets),
        address(clique.group_pairs),
        address(clique.tile_group_offsets),
        address(clique.parent_offsets),
        address(clique.members),
        address(clique.bucket_ids) + 4 * first,
        address(clique.coefficients),
        count,
        clique.num_tiles,
        clique.tile_rows,
        clique.orbitals,
        bucket,
        clique.edges_per_batch,
        bool(clique.distinct_rows),
        panel,
        flag,
        columns,
    )


def _cross_lbg(k, launch: SourceLaunch, clique: CliqueCross, geometry: tuple) -> None:


    from .aabb_lbg import RSL_ROWS_PER_JOB  # noqa: PLC0415

    plan, scratch, flag = (
        clique.lbg,
        address(clique.lbg_scratch),
        address(clique.lbg_flag),
    )
    _, n_a, _, n_b, _, _, _, n_b_src, _, _, _, _, stride, _ = launch.geometry
    num_pairs = int(clique.coefficients.size) // (int(clique.orbitals) ** 2)
    _check(
        lambda: k.launch_lbg_finite(
            launch.src_in, int(n_b_src) * int(stride), flag, launch.stream
        ),
        "lbg finiteness scan",
        launch,
    )
    _record(launch, "parent_start")
    if clique.residual_edges:
        if plan.rsl:
            for group in plan.rsl:
                m = group.members_per_clique
                block_doubles = num_pairs * (m * (m - 1) // 2) * 32
                for first in range(0, group.num_blocks, group.per_launch):
                    count = min(group.per_launch, group.num_blocks - first)
                    table = (
                        0
                        if group.coef_table is None
                        else address(group.coef_table) + 8 * first * block_doubles
                    )
                    _check(
                        lambda: k.launch_cross_rsl(
                            geometry,
                            launch.src_in,
                            launch.out,
                            address(group.block_first) + 4 * first,
                            address(group.block_count) + 4 * first,
                            address(group.members),
                            scratch,
                            address(clique.tasks),
                            address(clique.beta_words),
                            address(clique.coefficients),
                            table,
                            flag,
                            count,
                            RSL_ROWS_PER_JOB,
                            plan.rsl_panel_rows,
                            num_pairs,
                            clique.orbitals,
                            m,
                            launch.stream,
                        ),
                        "cross rsl",
                        launch,
                    )
        order = clique.residual_order
        _check(
            lambda: k.launch_cross_direct(
                geometry,
                launch.src_in,
                launch.out,
                offsets(clique.residual_off),
                address(clique.residual_lengths),
                address(clique.residual_pack),
                address(clique.beta_words),
                address(clique.tasks),
                address(launch.coefficients),
                launch.num_alpha * (launch.num_alpha - 1) // 2,
                n_b,
                launch.stream,
                address(order),
                0 if order is None else int(order.size),
                flag if plan.rsl else 0,
            ),
            "cross residual",
            launch,
        )
    _record(launch, "residual_end")
    for bk in plan.buckets:
        for first in range(0, bk.count, bk.per_launch):
            count = min(bk.per_launch, bk.count - first)
            panel_doubles = count * plan.panel_columns * bk.panel_row
            tables = _clique_tuple(
                clique,
                bk.bucket,
                bk.first + first,
                count,
                scratch,
                flag,
                plan.panel_columns,
            )
            _check(
                lambda: k.launch_cross_clique_pack(
                    geometry, launch.src_in, tables, launch.stream
                ),
                "cross lbg pack",
                launch,
            )
            _check(
                lambda: k.launch_cross_lbg(
                    geometry,
                    launch.src_in,
                    launch.out,
                    tables,
                    address(plan.edge_words),
                    address(plan.segment_begin),
                    address(plan.segment_end),
                    address(plan.segment_pair),
                    plan.num_segments,
                    scratch + 8 * panel_doubles,
                    bk.out_stride,
                    flag,
                    launch.stream,
                ),
                "cross lbg",
                launch,
            )

        fallback = _clique_tuple(
            clique, bk.bucket, bk.first, bk.count, scratch, flag, plan.panel_columns
        )
        _check(
            lambda: k.launch_cross_clique_gated(
                geometry, launch.src_in, launch.out, fallback, launch.stream
            ),
            "cross clique fallback",
            launch,
        )
    _record(launch, "parent_end")


def _cross_clique(k, launch: SourceLaunch, clique: CliqueCross) -> None:

    _, n_a, _, n_b, _, _, _, n_b_src, _, _, _, _, stride, _ = launch.geometry
    geometry = (0, n_a, 0, n_b, 0, stride, 0, n_b_src, 0, n_a, 0, n_b, stride)
    if clique.lbg is not None:
        _cross_lbg(k, launch, clique, geometry)
        return
    _record(launch, "parent_start")
    if clique.residual_edges:
        order = clique.residual_order
        _check(
            lambda: k.launch_cross_direct(
                geometry,
                launch.src_in,
                launch.out,
                offsets(clique.residual_off),
                address(clique.residual_lengths),
                address(clique.residual_pack),
                address(clique.beta_words),
                address(clique.tasks),
                address(launch.coefficients),
                launch.num_alpha * (launch.num_alpha - 1) // 2,
                n_b,
                launch.stream,
                address(order),
                0 if order is None else int(order.size),
            ),
            "cross residual",
            launch,
        )
    _record(launch, "residual_end")
    start = clique.bucket_start
    for bucket in range(len(start) - 1):
        count = int(start[bucket + 1]) - int(start[bucket])
        if count == 0:
            continue
        per = int(clique.panel_batch[bucket]) if clique.panel_batch else count
        for first in range(0, count, per):
            _check(
                lambda: k.launch_cross_clique(
                    geometry,
                    launch.src_in,
                    launch.out,
                    address(clique.edge_words),
                    address(clique.group_offsets),
                    address(clique.group_pairs),
                    address(clique.tile_group_offsets),
                    address(clique.parent_offsets),
                    address(clique.members),
                    address(clique.bucket_ids) + 4 * (int(start[bucket]) + first),
                    address(clique.coefficients),
                    min(per, count - first),
                    clique.num_tiles,
                    clique.tile_rows,
                    clique.orbitals,
                    bucket,
                    clique.edges_per_batch,
                    clique.transposed,
                    launch.stream,
                    clique.parent_group,
                    clique.mma,
                    clique.distinct_rows,
                    address(clique.panel),
                    address(clique.nonfinite),
                    clique.panel_columns,
                ),
                "cross clique",
                launch,
            )
    _record(launch, "parent_end")


def _cross(k, launch: SourceLaunch) -> None:

    if not _cross_ready(launch):
        return
    view = launch.view
    _record(launch, "source_start")
    if isinstance(launch.cross, CliqueCross):
        _cross_clique(k, launch, launch.cross)
    else:
        _check(
            lambda: k.launch_cross_sell(
                launch.geometry,
                launch.src_in,
                launch.out,
                launch.beta_slot,
                offsets(view["d_beta2_ptr"]),
                address(view["d_beta2_j"]),
                address(view["d_beta2_p0"]),
                address(view["d_beta2_p1"]),
                address(launch.beta_strings),
                address(launch.coefficients),
                offsets(view["d_alpha2_sell_off"]),
                address(view["d_alpha2_sell_len"]),
                address(view["d_alpha2_sell_pack"]),
                launch.num_alpha,
                launch.num_beta,
                launch.stream,
            ),
            "cross SELL",
            launch,
        )
    _record(launch, "source_end")


def launch_steady_arms(launch: SourceLaunch) -> None:


    from . import kernel_launch  # noqa: PLC0415

    _alpha_doubles(kernel_launch, launch)
    _alpha_singles(kernel_launch, launch)
    _beta_doubles(kernel_launch, launch)
    _beta_singles(kernel_launch, launch)
    _cross(kernel_launch, launch)


def launch_cross(launch: SourceLaunch) -> None:


    from . import kernel_launch  # noqa: PLC0415

    _cross(kernel_launch, launch)


def cross_ready(launch: SourceLaunch) -> bool:

    return _cross_ready(launch)


def build_source_caches(
    geometry: tuple,
    terms: tuple,
    view: dict,
    caches: dict,
    same_alpha: bool,
    same_beta: bool,
    alpha_slot: int,
    beta_slot: int,
    error: int,
    stream: int,
) -> None:


    from . import kernel_launch  # noqa: PLC0415

    if (
        same_beta
        and caches.get("aaaa") is not None
        and view.get("d_alpha4_sell_off") is not None
    ):
        _check(
            lambda: kernel_launch.build_alpha_doubles(
                geometry,
                terms,
                (
                    offsets(view["d_alpha4_sell_off"]),
                    address(view["d_alpha4_sell_len"]),
                    address(view["d_alpha4_sell_j"]),
                    *(address(view[f"d_alpha4_sell_p{k}"]) for k in range(4)),
                    alpha_slot,
                    beta_slot,
                    address(caches["aaaa"]),
                ),
                stream,
            ),
            "alpha doubles build",
        )
    if (
        same_beta
        and caches.get("aa_c0") is not None
        and caches.get("aa_cpb") is not None
    ):
        _check(
            lambda: kernel_launch.build_alpha_singles(
                geometry,
                terms,
                (
                    offsets(view["d_alpha2_ptr"]),
                    address(view["d_alpha2_j"]),
                    address(view["d_alpha2_p0"]),
                    address(view["d_alpha2_p1"]),
                    alpha_slot,
                    beta_slot,
                    address(caches["aa_c0"]),
                    address(caches["aa_cpb"]),
                    error,
                ),
                stream,
            ),
            "alpha singles build",
        )
    if same_alpha and caches.get("bbbb") is not None:
        _check(
            lambda: kernel_launch.build_beta_doubles(
                geometry,
                terms,
                (
                    offsets(view["d_beta4_ptr"]),
                    0,
                    address(view["d_beta4_j"]),
                    *(address(view[f"d_beta4_p{k}"]) for k in range(4)),
                    alpha_slot,
                    beta_slot,
                    address(caches["bbbb"]),
                ),
                stream,
            ),
            "beta doubles build",
        )
    if (
        same_alpha
        and caches.get("bb_c0") is not None
        and caches.get("bb_cpb") is not None
    ):
        _check(
            lambda: kernel_launch.build_beta_singles(
                geometry,
                terms,
                (
                    offsets(view["d_beta2_ptr"]),
                    address(view["d_beta2_j"]),
                    address(view["d_beta2_p0"]),
                    address(view["d_beta2_p1"]),
                    alpha_slot,
                    beta_slot,
                    address(caches["bb_c0"]),
                    address(caches["bb_cpb"]),
                    error,
                ),
                stream,
            ),
            "beta singles build",
        )
