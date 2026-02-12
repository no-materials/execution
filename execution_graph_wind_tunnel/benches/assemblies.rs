use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use execution_graph::{ExecutionGraph, NodeId};
use execution_tape::asm::{Asm, FunctionSig, ProgramBuilder};
use execution_tape::host::{
    AccessSink, Host, HostError, ResourceKeyRef, SigHash, ValueRef, sig_hash_slices,
};
use execution_tape::program::{HostSigId, ValueType};
use execution_tape::value::{FuncId, Value};
use execution_tape::verifier::VerifiedProgram;
use execution_tape::vm::Limits;

const DEFAULT_ASSEMBLY_SEED: u64 = 0xA55E_2026;
const DEFAULT_ASSEMBLY_COUNT: usize = 200;

const ROOT_INPUT_NAME: &str = "root";
const PARENT_INPUT_NAME: &str = "parent";
const LOCAL_INPUT_NAME: &str = "local";
const GEOMETRY_KEY_INPUT_NAME: &str = "geometry_key";
const MATERIAL_KEY_INPUT_NAME: &str = "material_key";
const COLOR_KEY_INPUT_NAME: &str = "color_key";
const OUTPUT_NAME: &str = "value";
const MATERIAL_OUTPUT_NAME: &str = "material";
const COLOR_OUTPUT_NAME: &str = "color";
const ROOT_PROGRAM_NAME: &str = "root";
const ROOT_FUNCTION_NAME: &str = "root_eval";
const CHILD_PROGRAM_NAME: &str = "child";
const CHILD_FUNCTION_NAME: &str = "child_eval";

const INPUT_MUTATION_STEP: i64 = 1_i64 << 20;
const ASSEMBLY_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;
const GEOMETRY_SCALE_KEY_SALT: u64 = 0x51CA_1E00_DA7A_C0DE;
const MATERIAL_ROUGHNESS_KEY_SALT: u64 = 0x9A44_E6C3_2E2D_32D5;
const COLOR_HUE_KEY_SALT: u64 = 0x2B27_9D47_44F8_19A1;

const EXPECTED_GRAPH_NODES_200: usize = 2_987;
const EXPECTED_GRAPH_EDGES_200: usize = 2_787;

#[derive(Debug, Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64: deterministic and cheap.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range_inclusive_usize(&mut self, min: usize, max: usize) -> usize {
        debug_assert!(min <= max);
        if min == max {
            return min;
        }
        let span = (max - min) + 1;
        min + (self.next_u64() as usize % span)
    }
}

#[derive(Debug, Clone, Copy)]
struct FlatAssemblyNode {
    id: u32,
    parent_id: Option<u32>,
    part_seed: u64,
    transform_seed: u64,
}

#[derive(Debug, Clone)]
struct AssemblyTree {
    root: AssemblyNode,
}

#[derive(Debug, Clone)]
struct AssemblyNode {
    id: u32,
    part_seed: u64,
    transform_seed: u64,
    children: Vec<AssemblyNode>,
}

impl AssemblyTree {
    fn generate(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed ^ 0xD1CE_BA5E_F00D_F00D);
        let target_depth = 3 + rng.range_inclusive_usize(0, 2) as u32;
        let mut next_id = 0_u32;
        let root = AssemblyNode::generate_recursive(&mut rng, 1, target_depth, &mut next_id);
        Self { root }
    }

    fn flatten_preorder(&self) -> Vec<FlatAssemblyNode> {
        let mut out = Vec::new();
        self.root.collect_flat_nodes(None, &mut out);
        out
    }
}

impl AssemblyNode {
    fn generate_recursive(
        rng: &mut DeterministicRng,
        current_depth: u32,
        target_depth: u32,
        next_id: &mut u32,
    ) -> Self {
        let id = *next_id;
        *next_id = next_id.saturating_add(1);

        let part_seed = rng.next_u64();
        let transform_seed = rng.next_u64();
        let children = if current_depth < target_depth {
            let max_children = if current_depth + 1 < target_depth {
                3
            } else {
                2
            };
            let child_count = rng.range_inclusive_usize(1, max_children);
            let mut out = Vec::with_capacity(child_count);
            for _ in 0..child_count {
                out.push(Self::generate_recursive(
                    rng,
                    current_depth + 1,
                    target_depth,
                    next_id,
                ));
            }
            out
        } else {
            Vec::new()
        };

        Self {
            id,
            part_seed,
            transform_seed,
            children,
        }
    }

    fn collect_flat_nodes(&self, parent_id: Option<u32>, out: &mut Vec<FlatAssemblyNode>) {
        out.push(FlatAssemblyNode {
            id: self.id,
            parent_id,
            part_seed: self.part_seed,
            transform_seed: self.transform_seed,
        });

        for child in &self.children {
            child.collect_flat_nodes(Some(self.id), out);
        }
    }
}

fn generate_assembly_tree(seed: u64) -> AssemblyTree {
    AssemblyTree::generate(seed)
}

#[derive(Debug, Clone)]
struct DemoHost {
    geometry_scale_sig: SigHash,
    material_roughness_sig: SigHash,
    color_hue_sig: SigHash,
    geometry_scale_state_values: Arc<Mutex<BTreeMap<u64, i64>>>,
    material_roughness_state_values: Arc<Mutex<BTreeMap<u64, i64>>>,
    color_hue_state_values: Arc<Mutex<BTreeMap<u64, i64>>>,
}

impl DemoHost {
    fn new(
        geometry_scale_sig: SigHash,
        material_roughness_sig: SigHash,
        color_hue_sig: SigHash,
        geometry_scale_state_values: Arc<Mutex<BTreeMap<u64, i64>>>,
        material_roughness_state_values: Arc<Mutex<BTreeMap<u64, i64>>>,
        color_hue_state_values: Arc<Mutex<BTreeMap<u64, i64>>>,
    ) -> Self {
        Self {
            geometry_scale_sig,
            material_roughness_sig,
            color_hue_sig,
            geometry_scale_state_values,
            material_roughness_state_values,
            color_hue_state_values,
        }
    }
}

impl Host for DemoHost {
    fn call(
        &mut self,
        symbol: &str,
        sig_hash: SigHash,
        args: &[ValueRef<'_>],
        rets: &mut [Value],
        access: Option<&mut dyn AccessSink>,
    ) -> Result<u64, HostError> {
        let [ValueRef::I64(raw_key)] = args else {
            return Err(HostError::Failed);
        };

        let (expected_sig, state_key, state_values, default_value_fn): (
            SigHash,
            u64,
            &Arc<Mutex<BTreeMap<u64, i64>>>,
            fn(u64) -> i64,
        ) = match symbol {
            "geometry.scale" => (
                self.geometry_scale_sig,
                geometry_scale_state_key(*raw_key as u64),
                &self.geometry_scale_state_values,
                geometry_scale_value,
            ),
            "material.roughness" => (
                self.material_roughness_sig,
                material_roughness_state_key(*raw_key as u64),
                &self.material_roughness_state_values,
                material_roughness_value,
            ),
            "color.hue" => (
                self.color_hue_sig,
                color_hue_state_key(*raw_key as u64),
                &self.color_hue_state_values,
                color_hue_value,
            ),
            _ => return Err(HostError::UnknownSymbol),
        };

        if sig_hash != expected_sig {
            return Err(HostError::SignatureMismatch);
        }

        if let Some(a) = access {
            a.read(ResourceKeyRef::HostState {
                op: sig_hash,
                key: state_key,
            });
        }

        let mut state_values = match state_values.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let value = *state_values
            .entry(state_key)
            .or_insert_with(|| default_value_fn(state_key));

        rets[0] = Value::I64(value);
        Ok(0)
    }
}

#[derive(Debug, Clone)]
struct ProgramTemplates {
    root_program: Arc<VerifiedProgram>,
    root_entry: FuncId,
    child_program: Arc<VerifiedProgram>,
    child_entry: FuncId,
}

impl ProgramTemplates {
    fn build() -> Self {
        let (root_program, root_entry) = build_root_program();
        let (child_program, child_entry) = build_child_program();
        Self {
            root_program: Arc::new(root_program),
            root_entry,
            child_program: Arc::new(child_program),
            child_entry,
        }
    }
}

#[derive(Debug)]
struct RootInvalidationWorkload {
    graph: ExecutionGraph<DemoHost>,
    root_input_bases: Vec<(NodeId, i64)>,
    total_nodes: usize,
    total_edges: usize,
}

fn build_workload() -> RootInvalidationWorkload {
    let geometry_scale_sig_hash = sig_hash_slices(&[ValueType::I64], &[ValueType::I64]);
    let material_roughness_sig_hash = sig_hash_slices(&[ValueType::I64], &[ValueType::I64]);
    let color_hue_sig_hash = sig_hash_slices(&[ValueType::I64], &[ValueType::I64]);

    let geometry_scale_state_values = Arc::new(Mutex::new(BTreeMap::new()));
    let material_roughness_state_values = Arc::new(Mutex::new(BTreeMap::new()));
    let color_hue_state_values = Arc::new(Mutex::new(BTreeMap::new()));

    let mut graph = ExecutionGraph::new(
        DemoHost::new(
            geometry_scale_sig_hash,
            material_roughness_sig_hash,
            color_hue_sig_hash,
            Arc::clone(&geometry_scale_state_values),
            Arc::clone(&material_roughness_state_values),
            Arc::clone(&color_hue_state_values),
        ),
        Limits::default(),
    );

    let templates = ProgramTemplates::build();

    let mut root_input_bases = Vec::new();
    let mut total_nodes = 0_usize;
    let mut total_edges = 0_usize;

    for assembly_index in 0..DEFAULT_ASSEMBLY_COUNT {
        let assembly_seed = seed_for_assembly(DEFAULT_ASSEMBLY_SEED, assembly_index);
        let assembly_tree = generate_assembly_tree(assembly_seed);
        let flat_nodes = assembly_tree.flatten_preorder();

        let mut graph_nodes_by_assembly_id: BTreeMap<u32, NodeId> = BTreeMap::new();

        for flat_node in flat_nodes {
            let local_value = seed_value(flat_node.transform_seed, flat_node.part_seed);
            let node_key = geometry_node_key(assembly_index, flat_node.id);

            let geometry_key_input_value = node_key as i64;
            let material_key_input_value = node_key as i64;
            let color_key_input_value = node_key as i64;

            let geometry_state_key = geometry_scale_state_key(node_key);
            let material_state_key = material_roughness_state_key(node_key);
            let color_state_key = color_hue_state_key(node_key);

            insert_geometry_state_value(&geometry_scale_state_values, geometry_state_key);
            insert_material_state_value(&material_roughness_state_values, material_state_key);
            insert_color_state_value(&color_hue_state_values, color_state_key);

            let graph_node = match flat_node.parent_id {
                None => {
                    let node = graph.add_node(
                        templates.root_program.clone(),
                        templates.root_entry,
                        vec![
                            ROOT_INPUT_NAME.into(),
                            GEOMETRY_KEY_INPUT_NAME.into(),
                            MATERIAL_KEY_INPUT_NAME.into(),
                            COLOR_KEY_INPUT_NAME.into(),
                        ],
                    );
                    graph.set_input_value(node, ROOT_INPUT_NAME, Value::I64(local_value));
                    graph.set_input_value(
                        node,
                        GEOMETRY_KEY_INPUT_NAME,
                        Value::I64(geometry_key_input_value),
                    );
                    graph.set_input_value(
                        node,
                        MATERIAL_KEY_INPUT_NAME,
                        Value::I64(material_key_input_value),
                    );
                    graph.set_input_value(
                        node,
                        COLOR_KEY_INPUT_NAME,
                        Value::I64(color_key_input_value),
                    );
                    root_input_bases.push((node, local_value));
                    node
                }
                Some(parent_id) => {
                    let parent_node = graph_nodes_by_assembly_id[&parent_id];
                    let node = graph.add_node(
                        templates.child_program.clone(),
                        templates.child_entry,
                        vec![
                            PARENT_INPUT_NAME.into(),
                            LOCAL_INPUT_NAME.into(),
                            GEOMETRY_KEY_INPUT_NAME.into(),
                            MATERIAL_KEY_INPUT_NAME.into(),
                            COLOR_KEY_INPUT_NAME.into(),
                        ],
                    );
                    graph.set_input_value(node, LOCAL_INPUT_NAME, Value::I64(local_value));
                    graph.set_input_value(
                        node,
                        GEOMETRY_KEY_INPUT_NAME,
                        Value::I64(geometry_key_input_value),
                    );
                    graph.set_input_value(
                        node,
                        MATERIAL_KEY_INPUT_NAME,
                        Value::I64(material_key_input_value),
                    );
                    graph.set_input_value(
                        node,
                        COLOR_KEY_INPUT_NAME,
                        Value::I64(color_key_input_value),
                    );
                    graph.connect(parent_node, OUTPUT_NAME, node, PARENT_INPUT_NAME);
                    total_edges = total_edges.saturating_add(1);
                    node
                }
            };

            graph_nodes_by_assembly_id.insert(flat_node.id, graph_node);
            total_nodes = total_nodes.saturating_add(1);
        }
    }

    assert_eq!(total_nodes, EXPECTED_GRAPH_NODES_200);
    assert_eq!(total_edges, EXPECTED_GRAPH_EDGES_200);

    RootInvalidationWorkload {
        graph,
        root_input_bases,
        total_nodes,
        total_edges,
    }
}

fn write_epoch_inputs(
    graph: &mut ExecutionGraph<DemoHost>,
    base_values: &[(NodeId, i64)],
    input_name: &str,
    epoch: u64,
) {
    for (index, (node, base_value)) in base_values.iter().copied().enumerate() {
        graph.set_input_value(
            node,
            input_name,
            Value::I64(epoch_mutated_input_value(base_value, epoch, index)),
        );
    }
}

fn epoch_mutated_input_value(base_value: i64, epoch: u64, input_index: usize) -> i64 {
    let hash = epoch
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((input_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .rotate_left(11);
    let phase = (hash & 0x1f) as i64 - 16;
    base_value.wrapping_add(phase.wrapping_mul(INPUT_MUTATION_STEP))
}

fn seed_for_assembly(base_seed: u64, assembly_index: usize) -> u64 {
    base_seed.wrapping_add((assembly_index as u64).wrapping_mul(ASSEMBLY_SEED_STRIDE))
}

fn seed_value(transform_seed: u64, part_seed: u64) -> i64 {
    let mixed = transform_seed
        ^ part_seed.rotate_left(17)
        ^ 0xA5A5_A5A5_5A5A_5A5A
        ^ transform_seed.rotate_left(7);
    (mixed as i64).wrapping_mul(31).wrapping_add(17)
}

fn geometry_node_key(assembly_index: usize, node_id: u32) -> u64 {
    ((assembly_index as u64) << 32) | u64::from(node_id)
}

fn geometry_scale_state_key(node_key: u64) -> u64 {
    node_key.rotate_left(19) ^ GEOMETRY_SCALE_KEY_SALT
}

fn material_roughness_state_key(node_key: u64) -> u64 {
    node_key.rotate_left(23) ^ MATERIAL_ROUGHNESS_KEY_SALT
}

fn color_hue_state_key(node_key: u64) -> u64 {
    node_key.rotate_left(29) ^ COLOR_HUE_KEY_SALT
}

fn geometry_scale_value(state_key: u64) -> i64 {
    let mixed = state_key.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(9);
    let centered = ((mixed >> 59) as i64) - 16;
    centered * 257
}

fn material_roughness_value(state_key: u64) -> i64 {
    let mixed = state_key.wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(7);
    let centered = ((mixed >> 56) as i64) - 128;
    centered * 33
}

fn color_hue_value(state_key: u64) -> i64 {
    let mixed = state_key
        .wrapping_mul(0xA24B_AED4_963E_E407)
        .rotate_left(13);
    let centered = ((mixed >> 57) as i64) - 64;
    centered * 41
}

fn insert_geometry_state_value(state_values: &Arc<Mutex<BTreeMap<u64, i64>>>, state_key: u64) {
    let mut state_values = match state_values.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    state_values
        .entry(state_key)
        .or_insert_with(|| geometry_scale_value(state_key));
}

fn insert_material_state_value(state_values: &Arc<Mutex<BTreeMap<u64, i64>>>, state_key: u64) {
    let mut state_values = match state_values.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    state_values
        .entry(state_key)
        .or_insert_with(|| material_roughness_value(state_key));
}

fn insert_color_state_value(state_values: &Arc<Mutex<BTreeMap<u64, i64>>>, state_key: u64) {
    let mut state_values = match state_values.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    state_values
        .entry(state_key)
        .or_insert_with(|| color_hue_value(state_key));
}

fn register_geometry_scale_host_sig(builder: &mut ProgramBuilder) -> HostSigId {
    builder.host_sig_for(
        "geometry.scale",
        execution_tape::host::HostSig {
            args: vec![ValueType::I64],
            rets: vec![ValueType::I64],
        },
    )
}

fn register_material_roughness_host_sig(builder: &mut ProgramBuilder) -> HostSigId {
    builder.host_sig_for(
        "material.roughness",
        execution_tape::host::HostSig {
            args: vec![ValueType::I64],
            rets: vec![ValueType::I64],
        },
    )
}

fn register_color_hue_host_sig(builder: &mut ProgramBuilder) -> HostSigId {
    builder.host_sig_for(
        "color.hue",
        execution_tape::host::HostSig {
            args: vec![ValueType::I64],
            rets: vec![ValueType::I64],
        },
    )
}

fn build_root_program() -> (VerifiedProgram, FuncId) {
    let mut builder = ProgramBuilder::new();
    builder.set_program_name(ROOT_PROGRAM_NAME);
    let geometry_scale_sig = register_geometry_scale_host_sig(&mut builder);
    let material_roughness_sig = register_material_roughness_host_sig(&mut builder);
    let color_hue_sig = register_color_hue_host_sig(&mut builder);

    let mut asm = Asm::new();
    asm.host_call(0, geometry_scale_sig, 0, &[2], &[5]);
    asm.host_call(0, material_roughness_sig, 0, &[3], &[6]);
    asm.host_call(0, color_hue_sig, 0, &[4], &[7]);
    asm.i64_add(8, 1, 5);
    asm.ret(0, &[8, 6, 7]);

    let entry = builder
        .push_function_checked(
            asm,
            FunctionSig {
                arg_types: vec![
                    ValueType::I64,
                    ValueType::I64,
                    ValueType::I64,
                    ValueType::I64,
                ],
                ret_types: vec![ValueType::I64, ValueType::I64, ValueType::I64],
                reg_count: 9,
            },
        )
        .unwrap();
    builder
        .set_function_name(entry, ROOT_FUNCTION_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 0, ROOT_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 1, GEOMETRY_KEY_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 2, MATERIAL_KEY_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 3, COLOR_KEY_INPUT_NAME)
        .unwrap();
    builder
        .set_function_output_name(entry, 0, OUTPUT_NAME)
        .unwrap();
    builder
        .set_function_output_name(entry, 1, MATERIAL_OUTPUT_NAME)
        .unwrap();
    builder
        .set_function_output_name(entry, 2, COLOR_OUTPUT_NAME)
        .unwrap();

    (builder.build_verified().unwrap(), entry)
}

fn build_child_program() -> (VerifiedProgram, FuncId) {
    let mut builder = ProgramBuilder::new();
    builder.set_program_name(CHILD_PROGRAM_NAME);
    let geometry_scale_sig = register_geometry_scale_host_sig(&mut builder);
    let material_roughness_sig = register_material_roughness_host_sig(&mut builder);
    let color_hue_sig = register_color_hue_host_sig(&mut builder);

    let mut asm = Asm::new();
    asm.host_call(0, geometry_scale_sig, 0, &[3], &[6]);
    asm.host_call(0, material_roughness_sig, 0, &[4], &[7]);
    asm.host_call(0, color_hue_sig, 0, &[5], &[8]);
    asm.i64_add(9, 1, 2);
    asm.i64_add(10, 9, 6);
    asm.ret(0, &[10, 7, 8]);

    let entry = builder
        .push_function_checked(
            asm,
            FunctionSig {
                arg_types: vec![
                    ValueType::I64,
                    ValueType::I64,
                    ValueType::I64,
                    ValueType::I64,
                    ValueType::I64,
                ],
                ret_types: vec![ValueType::I64, ValueType::I64, ValueType::I64],
                reg_count: 11,
            },
        )
        .unwrap();
    builder
        .set_function_name(entry, CHILD_FUNCTION_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 0, PARENT_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 1, LOCAL_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 2, GEOMETRY_KEY_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 3, MATERIAL_KEY_INPUT_NAME)
        .unwrap();
    builder
        .set_function_input_name(entry, 4, COLOR_KEY_INPUT_NAME)
        .unwrap();
    builder
        .set_function_output_name(entry, 0, OUTPUT_NAME)
        .unwrap();
    builder
        .set_function_output_name(entry, 1, MATERIAL_OUTPUT_NAME)
        .unwrap();
    builder
        .set_function_output_name(entry, 2, COLOR_OUTPUT_NAME)
        .unwrap();

    (builder.build_verified().unwrap(), entry)
}

fn bench_assemblies_root_invalidation_200(c: &mut Criterion) {
    let mut workload = build_workload();

    // Warm up dependency tracking exactly like the GUI does before invalidation interactions.
    let warm = workload.graph.run_all().unwrap();
    assert_eq!(warm.executed_nodes, workload.total_nodes);

    // Sanity-check one root invalidation run before benchmarking.
    let mut epoch = 1_u64;
    write_epoch_inputs(
        &mut workload.graph,
        &workload.root_input_bases,
        ROOT_INPUT_NAME,
        epoch,
    );
    workload.graph.invalidate_input(ROOT_INPUT_NAME);
    let check = workload.graph.run_all().unwrap();
    assert_eq!(check.executed_nodes, workload.total_nodes);
    assert_eq!(workload.total_edges, EXPECTED_GRAPH_EDGES_200);

    c.bench_function("nextgen_root_invalidation_200", |b| {
        b.iter(|| {
            epoch = epoch.wrapping_add(1);
            write_epoch_inputs(
                &mut workload.graph,
                &workload.root_input_bases,
                ROOT_INPUT_NAME,
                epoch,
            );
            workload.graph.invalidate_input(ROOT_INPUT_NAME);
            let summary = workload.graph.run_all().unwrap();
            black_box(summary.executed_nodes);
        });
    });
}

criterion_group!(benches, bench_assemblies_root_invalidation_200);
criterion_main!(benches);
