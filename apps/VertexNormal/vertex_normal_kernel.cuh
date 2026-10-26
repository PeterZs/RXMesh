#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.h"

#include <glm/gtx/norm.hpp>

/**
 * vertex_normal()
 */
template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_normal(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                             rxmesh::VertexAttribute<T> normals)
{
    using namespace rxmesh;
    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // get the face's three vertices coordinates
        const VertexHandle vertices[3] = {fv[0], fv[1], fv[2]};
        const vec3<T>      c0          = coords.template to_glm<3>(vertices[0]);
        const vec3<T>      c1          = coords.template to_glm<3>(vertices[1]);
        const vec3<T>      c2          = coords.template to_glm<3>(vertices[2]);

        // compute the face normal
        vec3<T> n = cross(c1 - c0, c2 - c0);

        // the three edges length
        vec3<T> l(glm::distance2(c0, c1),
                  glm::distance2(c1, c2),
                  glm::distance2(c2, c0));

        // add the face's normal to its vertices
#pragma unroll
        for (uint32_t v = 0; v < 3; ++v) {
            // Reuse the reciprocal and resolve addresses before atomic writes.
            const T  inverse = T(1) / (l[v] + l[(v + 2) % 3]);
            T* const nx      = &normals(vertices[v], 0);
            T* const ny      = &normals(vertices[v], 1);
            T* const nz      = &normals(vertices[v], 2);
            atomicAdd(nx, n[0] * inverse);
            atomicAdd(ny, n[1] * inverse);
            atomicAdd(nz, n[2] * inverse);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, vn_lambda);
}