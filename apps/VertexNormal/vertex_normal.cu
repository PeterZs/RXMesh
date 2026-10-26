// Compute the vertex normal according to
// Max, Nelson. "Weights for computing vertex normals from facet normals."
// Journal of Graphics Tools 4, no. 2 (1999): 1-6.

#include <CLI/CLI.hpp>
#include <cmath>
#include <cstdlib>

#include "rxmesh/attribute.h"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "vertex_normal_kernel.cuh"
#include "vertex_normal_ref.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    num_run       = 100;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "vertex_normal_hardwired.cuh"

template <typename T>
void vertex_normal_rxmesh(rxmesh::RXMeshStatic&              rx,
                          const std::vector<std::vector<T>>& Verts,
                          const std::vector<T>&              vertex_normal_gold)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("VertexNormal_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    auto coords = rx.add_vertex_attribute<T>(Verts, "coordinates", SoA);
    auto v_normals =
        rx.add_vertex_attribute<T>("v_normals", 3, LOCATION_ALL, SoA);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({rxmesh::Op::FV},
                          launch_box,
                          (void*)compute_vertex_normal<T, blockThreads>);


    TestData td;
    td.test_name   = "VertexNormal";
    td.num_threads = launch_box.num_threads;
    td.num_blocks  = launch_box.blocks;
    td.dyn_smem    = launch_box.smem_bytes_dyn;
    td.static_smem = launch_box.smem_bytes_static;
    td.num_reg     = launch_box.num_registers_per_thread;

    auto launch = [&]() {
        compute_vertex_normal<T, blockThreads><<<launch_box.blocks,
                                                 launch_box.num_threads,
                                                 launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *v_normals);
        CUDA_ERROR(cudaGetLastError());
    };

    GPUTimer timer;
    warmup_gpu(1000.0f);

    float vn_time = 0;
    for (uint32_t itr = 0; itr < Arg.num_run; ++itr) {
        // Time only the kernel; finish the reset before recording the start.
        v_normals->reset(T(0), DEVICE);
        CUDA_ERROR(cudaStreamSynchronize(nullptr));
        timer.start();
        launch();
        timer.stop();
        const float elapsed = timer.elapsed_millis();
        td.time_ms.push_back(elapsed);
        vn_time += elapsed;
    }

    RXMESH_INFO("vertex_normal_rxmesh() vertex normal kernel took {} (ms)",
                vn_time / Arg.num_run);


    v_normals->move(DEVICE, HOST);
    CUDA_ERROR(cudaStreamSynchronize(nullptr));

    bool         passed = true;
    const double atol   = sizeof(T) == 8 ? 1e-9 : 1e-4;
    const double rtol   = sizeof(T) == 8 ? 1e-8 : 1e-3;
    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            uint32_t v_id = rx.map_to_global(vh);

            for (uint32_t i = 0; i < 3; ++i) {
                const double ref = double(vertex_normal_gold[v_id * 3 + i]);
                const double val = double((*v_normals)(vh, i));
                if (!std::isfinite(ref) || !std::isfinite(val) ||
                    std::abs(ref - val) > atol + rtol * std::abs(ref)) {
                    passed = false;
                }
            }
        },
        nullptr,
        false);
    td.passed.push_back(passed);

    if (!passed) {
        RXMESH_ERROR("VertexNormal RXMesh validation failed");
    }

    // Finalize report
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "VertexNormal_RXMesh_" + extract_file_name(Arg.obj_file_name));

#ifdef USE_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexVectorQuantity("v_normals", *v_normals);
#endif
}

int vertex_normal_main()
{
    using namespace rxmesh;

    // Load mesh
    std::vector<std::vector<rx_coord_t>> Verts;
    std::vector<std::vector<uint32_t>>   Faces;

    if (!import_obj(Arg.obj_file_name, Verts, Faces)) {
        RXMESH_ERROR("Failed to import OBJ file: {}", Arg.obj_file_name);
        return EXIT_FAILURE;
    }

    RXMeshStatic rx(Faces);

#ifdef USE_POLYSCOPE
    rx.add_vertex_coordinates(Verts);
#endif

    // Serial reference
    std::vector<rx_coord_t> vertex_normal_gold(3 * Verts.size());
    vertex_normal_ref(Faces, Verts, vertex_normal_gold);

    vertex_normal_rxmesh(rx, Verts, vertex_normal_gold);

    vertex_normal_hardwired(Faces, Verts, vertex_normal_gold);

#ifdef USE_POLYSCOPE
    polyscope::show();
#endif

    return 0;
}

int main(int argc, char** argv)
{
    using namespace rxmesh;

    CLI::App app{"VertexNormal - Vertex-normal computation benchmark"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("--num_run",
                   Arg.num_run,
                   "Number of iterations for performance testing")
        ->default_val(100u)
        ->check(CLI::PositiveNumber);

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(static_cast<int>(Arg.device_id));

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("num_run= {}", Arg.num_run);
    RXMESH_INFO("device_id= {}", Arg.device_id);

    return vertex_normal_main();
}