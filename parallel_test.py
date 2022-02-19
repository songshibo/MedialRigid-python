import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, kernel_profiler=True)

A_np = np.random.rand(1000, 3)
B_np = np.random.rand(1000, 3)
C_np = np.random.rand(1000, 3)

index_range = np.array([[0, 999], [1000, 1999], [2000, 2999]]).astype(np.uint8)

total = np.concatenate((A_np, B_np, C_np), axis=0).astype(np.float32)

T = ti.field(ti.f32)
ti.root.dense(ti.ij, total.shape).place(T)
T.from_numpy(total)

idx_range = ti.field(ti.uint8)
ti.root.dense(ti.ij, (3, 2)).place(idx_range)
idx_range.from_numpy(index_range)


@ti.kernel
def test1():
    valid_num = 0
    for x, y in ti.ndrange((0, 3000), (0, 3000)):
        in_lower_area = 0
        for i in ti.static(range(3)):
            if (x >= idx_range[i, 0] and x <= idx_range[i, 1] and y >= idx_range[i, 0] and y <= idx_range[i, 1]) or (x >= y):
                in_lower_area += 1
        if in_lower_area == 0:
            valid_num += 1
    print(valid_num)


test1()
ti.print_kernel_profile_info()
