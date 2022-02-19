import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, kernel_profiler=True)

A_np = np.random.rand(1000, 3)
B_np = np.random.rand(1000, 3)
C_np = np.random.rand(1000, 3)

index_range = np.array([[0, 999], [1000, 1999], [2000, 2999]]).astype(np.int32)

total = np.concatenate((A_np, B_np, C_np), axis=0).astype(np.float32)

T = ti.field(ti.f32)
ti.root.dense(ti.ij, total.shape).place(T)

idx_range = ti.field(ti.int32)
ti.root.dense(ti.ij, (3, 2)).place(idx_range)

T.from_numpy(total)
idx_range.from_numpy(index_range)


@ti.kernel
def test1():
    valid_num = 0
    for x, y in ti.ndrange(3000, 3000):
        in_ignore_area = (x >= y)
        for i in ti.static(range(3)):
            low = idx_range[i, 0]
            high = idx_range[i, 1]
            in_area = (low <= x <= high) and (low <= y <= high)
            in_ignore_area = in_ignore_area or in_area
        if not in_ignore_area:
            ti.atomic_add(valid_num, 1)
            # break
    print(valid_num)


test1()
ti.print_kernel_profile_info()
