# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:47:02 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from tqdm import tqdm  
from scipy.ndimage import zoom

while True:
    print("############  Menu:  ############")
    print("1. 高斯滤波")
    print("2. 中值滤波")
    print("3. 均值滤波")
    print("4. Motion滤波")
    print("5. MAX-MIN滤波")
    print("6. 差分滤波")
    print("7. Sobel滤波")
    print("8. Prewitt滤波")
    print("9. Laplacian滤波")
    print("10. Emboss滤波")
    print("11. LoG滤波")
    print("0. Exit")


    L1_option = int(input("Enter the L1_option (0/1/2/3/.../11): "))

    if L1_option == 0:
        print("Exiting the program.")
        break  # Exit the loop and end the program
    if L1_option == 1:
       
            # 创建高斯核的函数，sigma越小,核函数更密集更陡峭,相当于更局部的平滑滤波
            def gaussian_kernel(size, sigma=1.3):
                size = int(size) // 2
                x, y = np.mgrid[-size:size + 1, -size:size + 1]
                normal = 1 / (2.0 * np.pi * sigma**2)
                g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
                return g
            
            # 将高斯滤波器应用于图像的函数
            def gaussian_filter(image, kernel):
                # 填充图像，防止核函数移动到图像边缘时,出现像素溢出边界的情况
                pad_height = kernel.shape[0] // 2
                pad_width = kernel.shape[1] // 2
                padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
            
                # 使用卷积运算以加快计算速度
                new_image = np.zeros_like(image, dtype=np.float32)
            
                # 使用tqdm显示进度条
                for i in tqdm(range(image.shape[0]), desc="Processing", position=0, leave=True):
                    for j in range(image.shape[1]):
                        new_image[i, j] = np.sum(kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]])
            
                return new_image
            
            # 使用gdal读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # 将图像下采样
            downsampled_image = image[::4, ::4]
            
            # 创建高斯核
            kernel = gaussian_kernel(3, 2)
            
            # 将高斯滤波器应用于下采样图像
            filtered_downsampled_image = gaussian_filter(downsampled_image, kernel)
            
            # 使用双线性插值对滤波后的图像进行上采样
            upsampled_image = zoom(filtered_downsampled_image, 4, order=1)
                    
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("Gaussian Filter Image")
            plt.imshow(upsampled_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("Gaussian Filter Histogram")
            plt.hist(upsampled_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
    
    if L1_option == 2:
            # 应用中值滤波器到图像的函数，替换像素值为极端亮度或暗度的噪点,有效修复此类噪声
            def median_filter_optimized(image, kernel_size=3):
                pad = kernel_size // 2
                padded_image = np.pad(image, pad, mode='constant', constant_values=0)
                median_image = np.zeros_like(image, dtype=np.uint8)
            
                # 创建一个数组来存储邻域内的像素值
                neighborhood = np.zeros(kernel_size * kernel_size, dtype=np.uint8)
            
                # 迭代图像
                total_iterations = image.shape[0] * image.shape[1]
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            # 提取邻域内的像素值
                            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size].ravel()
                            # 使用QuickSelect算法找到中值
                            median_value = np.median(neighborhood)
                            median_image[i, j] = median_value
                            pbar.update(1)  # 更新进度条
            
                return median_image
            
            # 使用gdal读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            
            # 对下采样图像应用优化的中值滤波器
            median_filtered_downsampled_image = median_filter_optimized(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            upsampled_image = zoom(median_filtered_downsampled_image, 4, order=1)
               
            
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("median_filtered_image")
            plt.imshow(upsampled_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("median Filter Histogram")
            plt.hist(upsampled_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
    
    if L1_option == 3:
            # 创建均值滤波器（3x3）的函数，平滑图像,抑制噪声。利用像素方围区域的平均亮度替代中心像素的亮度,可以有效抑制高频噪声。
            def mean_filter(image, kernel_size=3):

                # 计算填充大小
                pad = kernel_size // 2
                # 创建图像的填充版本
                padded_image = np.pad(image, pad, mode='constant')
                # 用零初始化输出图像
                filtered_image = np.zeros_like(image)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用tqdm显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 应用均值滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            # 提取感兴趣区域
                            region = padded_image[i:i + kernel_size, j:j + kernel_size]
                            # 计算均值并将其分配给过滤后的图像
                            filtered_image[i, j] = np.mean(region)
                            pbar.update(1)  # 更新进度条
            
                return filtered_image.astype(np.uint8)
            
            # 使用gdal读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            
            # 对下采样图像应用均值滤波器
            mean_filtered_downsampled_image = mean_filter(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            upsampled_image = zoom(mean_filtered_downsampled_image, 4, order=1)

            
            
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("mean_filtered_image")
            plt.imshow(upsampled_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("mean Filter Histogram")
            plt.hist(upsampled_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
    

    if L1_option == 4:
            # 运动模糊滤波的函数，利用相邻帧间类似像素进行多帧融合,可有效抑制随机噪声
            def motion_filter(image, kernel_size=3):
                # 将核初始化为全零
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                # 在对角线上填充1/3（与图像中的核相符），仅计算窗口矩阵的主对角线元素的均值
                for i in range(kernel_size):
                    kernel[i, i] = 1.0 / kernel_size
            
                # 填充图像
                pad = kernel_size // 2
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
                # 创建与输入图像相同大小的全零输出图像
                output = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用tqdm显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 应用运动滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            # 将核应用于邻域
                            output[i, j] = np.sum(kernel * padded_image[i:i + kernel_size, j:j + kernel_size])
                            pbar.update(1)  # 更新进度条
            
                return output.astype(np.uint8)
            
            # 使用gdal读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            
            # 应用运动滤波器
            motion_filtered_image = motion_filter(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            upsampled_image = zoom(motion_filtered_image, 4, order=1)

            
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("motion_filtered_image")
            plt.imshow(upsampled_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("motion Filter Histogram")
            plt.hist(upsampled_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
    
    if L1_option == 5:

            # MAX-MIN 滤波器的函数，将窗口中的最大值减去最小值赋给中心值，从而保持轮廓和边缘
            def max_min_filter(image, kernel_size=3):
                # 填充图像
                pad = kernel_size // 2
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)
                # 创建与输入图像相同大小的全零输出图像
                output = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用tqdm显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用 MAX-MIN 滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            # 提取邻域
                            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
                            # 计算最大和最小值
                            max_val = np.max(neighborhood)
                            min_val = np.min(neighborhood)
                            # 将差异赋值给输出图像
                            output[i, j] = max_val - min_val
                            pbar.update(1)  # 更新进度条
            
                return output.astype(np.uint8)
            
            # 使用gdal读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            
            # 应用 MAX-MIN 滤波器
            max_min_filtered_image = max_min_filter(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            max_min_filtered_image = zoom(max_min_filtered_image, 4, order=1)

            
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("max_min_filtered_image")
            plt.imshow(max_min_filtered_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("max_min Filter Histogram")
            plt.hist(max_min_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
    

    if L1_option == 6:

            # 定义差分滤波器函数
            def differential_filter(image):
                # 定义垂直和水平差分滤波器
                vertical_kernel = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
                horizontal_kernel = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
            
                pad = 1
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
                output_vertical = np.zeros_like(image, dtype=np.float32)
                output_horizontal = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用 tqdm 显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用差分滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            output_vertical[i, j] = np.sum(vertical_kernel * padded_image[i:i + 3, j:j + 3])
                            output_horizontal[i, j] = np.sum(horizontal_kernel * padded_image[i:i + 3, j:j + 3])
                            pbar.update(1)  # 更新进度条
            
                # 取绝对值，因为差异可以为负值
                output_vertical = np.abs(output_vertical).astype(np.uint8)
                output_horizontal = np.abs(output_horizontal).astype(np.uint8)
                return output_vertical, output_horizontal
            
            # 使用 gdal 读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            # 应用微分滤波器
            vertical_filtered_image, horizontal_filtered_image = differential_filter(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            vertical_filtered_image = zoom(vertical_filtered_image, 4, order=1)
            horizontal_filtered_image = zoom(horizontal_filtered_image, 4, order=1)


            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 3, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            
            plt.subplot(2, 3, 2)
            plt.title("Vertical Filtered Image")
            plt.imshow(vertical_filtered_image, cmap='gray')
            
            plt.subplot(2, 3, 3)
            plt.title("Horizontal Filtered Image")
            plt.imshow(horizontal_filtered_image, cmap='gray')
            
            plt.subplot(2, 3, 4)
            plt.title("OriginalHistogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            
            plt.subplot(2, 3, 5)
            plt.title("Vertical Filtered Histogram")
            plt.hist(vertical_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 3, 6)
            plt.title("Horizontal Filtered Histogram")
            plt.hist(horizontal_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.tight_layout()
            plt.show()
            

    if L1_option == 7:
    
            # 定义 Sobel 滤波器函数
            def sobel_filter(image):
                # 定义垂直和水平方向的 Sobel 滤波器
                vertical_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                horizontal_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            
                pad = 1
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
                output_vertical = np.zeros_like(image, dtype=np.float32)
                output_horizontal = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用 tqdm 显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用 Sobel 滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            output_vertical[i, j] = np.sum(vertical_kernel * padded_image[i:i + 3, j:j + 3])
                            output_horizontal[i, j] = np.sum(horizontal_kernel * padded_image[i:i + 3, j:j + 3])
                            pbar.update(1)  # 更新进度条
            
                # 组合垂直和水平方向的结果，计算Sobel的范数
                output_magnitude = np.sqrt(output_vertical**2 + output_horizontal**2)
            
                # 取绝对值，因为差异可以为负值
                output_magnitude = np.abs(output_magnitude).astype(np.uint8)
                return output_magnitude
            
            # 使用 gdal 读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            # 应用 Sobel 滤波器
            sobel_filtered_image = sobel_filter(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            sobel_filtered_image = zoom(sobel_filtered_image, 4, order=1)


            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("Sobel Filtered Image")
            plt.imshow(sobel_filtered_image, cmap='gray')
            
            plt.subplot(2, 2, 4)
            plt.title("Sobel Filtered Histogram")
            plt.hist(sobel_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.tight_layout()
            plt.show()
                    

    if L1_option == 8:

            def prewitt_filter(image):
                # 定义垂直和水平方向的 Prewitt 滤波器
                prewitt_vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
                prewitt_horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            
                pad = prewitt_vertical.shape[0] // 2
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
                output_v = np.zeros_like(image, dtype=np.float32)
                output_h = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用 tqdm 显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            output_v[i, j] = np.sum(prewitt_vertical * padded_image[i:i + prewitt_vertical.shape[0], j:j + prewitt_vertical.shape[1]])
                            output_h[i, j] = np.sum(prewitt_horizontal * padded_image[i:i + prewitt_horizontal.shape[0], j:j + prewitt_horizontal.shape[1]])
                            pbar.update(1)  # 更新进度条
            
                # 计算 Prewitt 滤波器的范数
                prewitt_filtered_image = np.sqrt(output_v**2 + output_h**2)
            
                # 取绝对值，因为差异可以为负值
                prewitt_filtered_image = np.abs(prewitt_filtered_image).astype(np.uint8)
                
                return prewitt_filtered_image

            
            # 使用 gdal 读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            downsampled_image = image[::4, ::4]
            
            # 应用 Prewitt 滤波器
            prewitt_filtered_image = prewitt_filter(downsampled_image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            prewitt_filtered_image = zoom(prewitt_filtered_image, 4, order=1)

                    
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("Prewitt Filtered Image")
            plt.imshow(prewitt_filtered_image, cmap='gray')
            
            plt.subplot(2, 2, 4)
            plt.title("Prewitt Filtered Histogram")
            plt.hist(prewitt_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.tight_layout()
            plt.show()
    

    if L1_option == 9:

            def laplacian_filter(image):
                # 定义拉普拉斯滤波器，四邻域模板
                laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            
                pad = laplacian.shape[0] // 2
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
                output = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用 tqdm 显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用拉普拉斯滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            output[i, j] = np.sum(laplacian * padded_image[i:i + laplacian.shape[0], j:j + laplacian.shape[1]])
                            pbar.update(1)  # 更新进度条
            
                # 将输出归一化到 0 到 255 的范围内
                output = np.clip(output, 0, 255).astype(np.uint8)
                
                return output

            
            # 使用 gdal 读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            image = image[::2, ::2]
            
            # 应用拉普拉斯滤波器
            laplacian_filtered_image = laplacian_filter(image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            laplacian_filtered_image = zoom(laplacian_filtered_image, 2, order=1)

            
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("laplacian_filtered_image")
            plt.imshow( laplacian_filtered_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("laplacian Filter Histogram")
            plt.hist( laplacian_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()


    if L1_option == 10:
            
            def emboss_filter(image):
                # 定义浮雕（emboss）滤波器
                emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
            
                pad = emboss.shape[0] // 2
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
                output = np.zeros_like(image, dtype=np.float32)
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用 tqdm 显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用浮雕滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            output[i, j] = np.sum(emboss * padded_image[i:i + emboss.shape[0], j:j + emboss.shape[1]])
                            pbar.update(1)  # 更新进度条
            
                # 将输出归一化到 0 到 255 的范围内
                output = np.clip(output, 0, 255).astype(np.uint8)
                
                return output

            
            # 使用 gdal 读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/2 x N/2 的降采样图像
            image = image[::4, ::4]
            
            # 应用浮雕滤波器
            emboss_filtered_image = emboss_filter(image)
            
            # 使用双线性插值对滤波后的图像进行上采样
            emboss_filtered_image = zoom(emboss_filtered_image, 4, order=1)
                    
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("Emboss_filtered_image")
            plt.imshow( emboss_filtered_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("Emboss Filter Histogram")
            plt.hist( emboss_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()

    if L1_option == 11:
             #使用高斯滤波器使图像平滑化之后再使用拉普拉斯滤波器使图像的轮廓更加清晰           
            def log_filter(image, size, sigma):
                # 计算填充和输出数组
                pad = size // 2
                padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
                output = np.zeros_like(image, dtype=np.float32)
            
                # LoG Kernel的尺寸和初始化
                K_size = size
                K = np.zeros((K_size, K_size), dtype=np.float32)
            
                # 生成LoG核
                for x in range(-pad, -pad + K_size):
                    for y in range(-pad, -pad + K_size):
                        # 核内每个元素的计算
                        K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
                
                # 将核归一化以确保总和为 0
                K /= (2 * np.pi * (sigma ** 6))
                K /= K.sum()
            
                # 计算总迭代次数
                total_iterations = image.shape[0] * image.shape[1]
            
                # 使用 tqdm 显示进度条
                with tqdm(total=total_iterations, desc="Processing", position=0, leave=True) as pbar:
                    # 对每个像素应用 LoG 滤波器
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            output[i, j] = np.sum(K * padded_image[i:i + size, j:j + size])
                            pbar.update(1)  # 更新进度条
            
                # 将输出归一化到 0 到 255 的范围内
                output = np.clip(output, 0, 255).astype(np.uint8)
            
                return output

            

            
            # 使用 gdal 读取图像
            file_path = 'IMAGERY.TIF'
            source_ds = gdal.Open(file_path)
            image = source_ds.GetRasterBand(1).ReadAsArray()
            
            # 验证图像是否正确加载
            if image is None:
                raise ValueError("检查图像文件的路径或确保文件存在")
            
            # M/4 x N/4 的降采样图像
            image = image[::4, ::4]
            
            # 应用 LoG 滤波器
            log_filtered_image = log_filter(image, 5,3)
            
            # 使用双线性插值对滤波后的图像进行上采样
            log_filtered_image = zoom(log_filtered_image, 4, order=1)

            
            # 显示原始图像和直方图
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            
            plt.subplot(2, 2, 3)
            plt.title("Log_filtered_image")
            plt.imshow( log_filtered_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("Log Filter Histogram")
            plt.hist( log_filtered_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()

















