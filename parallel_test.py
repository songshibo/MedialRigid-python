import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, kernel_profiler=True)

A_np = np.random.rand(1000, 3)
B_np = np.random.rand(2000, 3)
C_np = np.random.rand(1500, 3)

index_range = np.array(
    [[0, 1000], [1000, 3000], [3000, 4500]]).astype(np.int32)

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
    ti.block_dim(512)
    for x, y in ti.ndrange(4500, 4500):
        in_ignore_area = (x >= y)
        for i in ti.static(range(3)):
            low = idx_range[i, 0]
            high = idx_range[i, 1]
            in_area = (low <= x < high) and (low <= y < high)
            in_ignore_area = in_ignore_area or in_area
        if not in_ignore_area:
            ti.atomic_add(valid_num, 1)
            # break
    print(valid_num)


@ti.kernel
def test2(x_l: ti.int32, x_h: ti.int32, y_l: ti.int32, y_h: ti.int32):
    valid_num = 0
    ti.block_dim(512)
    for x, y in ti.ndrange((x_l, x_h), (y_l, y_h)):
        ti.atomic_add(valid_num, 1)
    print(valid_num)


# test1()
# test2(0, 1000, 1000, 4500)
# test2(1000, 3000, 3000, 4500)
for i in range(2):
    for j in range(i+1, 3):
        test2(idx_range[i, 0], idx_range[i, 1],
              idx_range[j, 0], idx_range[j, 1])
ti.print_kernel_profile_info()
