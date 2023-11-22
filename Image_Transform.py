# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:11:44 2023

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import zoom
from tqdm import tqdm  


while True:
    print("############  Menu:  ############")
    print("1. HSV变换")
    print("2. 傅里叶变换")
    print("3. PCA变换")
    print("0. Exit")

    L1_option = int(input("Enter the L1_option (0/1/2/3): "))
    
    

    if L1_option == 0:
        print("Exiting the program.")
        break  
    if L1_option == 1:
        def rgb_to_hsv(rgb_img):
            #H（Hue）色度：就是平常所说的颜色名称，如红色、蓝色、绿色
            #S（Saturation）饱和度：是指色彩的纯度，饱和度越低则颜色越黯淡，0≤S≤1。
            #V（Value）明度：表示了颜色的亮暗程度，坐标原点值为0，在锥体最上方的顶点上的值为1
            # 将RGB值归一化到范围[0, 1]
            normalized_img = rgb_img.astype(np.float32) / 255.0
            
            # 提取RGB通道
            r, g, b = normalized_img[:, :, 0], normalized_img[:, :, 1], normalized_img[:, :, 2]
            
            # 计算值（V）
            v = np.max(normalized_img, axis=2)
            
            # 计算饱和度（S）
            s = np.where(v != 0, (v - np.min(normalized_img, axis=2)) / v, 0)
            
            # 计算色调（H）
            h = np.zeros_like(v)
            
            idx = (v == r) & (v - np.min(normalized_img, axis=2) != 0)
            h[idx] = (60 * (g[idx] - b[idx]) / (v[idx] - np.min(normalized_img, axis=2)[idx]) + 360) % 360
            
            idx = (v == g) & (v - np.min(normalized_img, axis=2) != 0)
            h[idx] = (60 * (b[idx] - r[idx]) / (v[idx] - np.min(normalized_img, axis=2)[idx]) + 120) % 360
            
            idx = (v == b) & (v - np.min(normalized_img, axis=2) != 0)
            h[idx] = (60 * (r[idx] - g[idx]) / (v[idx] - np.min(normalized_img, axis=2)[idx]) + 240) % 360
        
            # 将HSV通道堆叠起来
            hsv_img = np.stack((h, s, v), axis=-1)
            
            # 将色调H通道的范围为[0,360]转换到范围[0, 1]
            hsv_img[:, :, 0] /= 360.0
            
            return hsv_img
        
        
        def hsv_transform(input_image_path):
            # 使用gdal读取输入的RGB图像
            ds = gdal.Open(input_image_path)
            rgb_img = np.transpose(ds.ReadAsArray(), (1, 2, 0))  # 假设图像的维度为（波段数，高度，宽度）
            
            # 执行HSV变换
            hsv_img = rgb_to_hsv(rgb_img)
            
            # 显示原始图像和变换后的图像
            plt.figure(figsize=(10, 5))
        
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_img)
            plt.title('Original RGB Image')
            plt.axis('off')
        
            plt.subplot(1, 2, 2)
            plt.imshow(hsv_img)
            plt.title('HSV Transformed Image')
            plt.axis('off')
        
            plt.show()
        
        if __name__ == "__main__":
            input_image_path = 'IMAGERY.TIF'  # 将此路径更改为您的GeoTIFF图像的路径
            hsv_transform(input_image_path)

    
    if L1_option == 2:
        def fft2(image):
            # 计算2D傅里叶变换
            rows, cols = image.shape
            #存储傅里叶变换的结果
            result = np.zeros((rows, cols), dtype=np.complex128)
        
            for u in range(rows):
                for v in range(cols):
                    # 使用傅里叶变换的定义计算结果
                    result[u, v] = np.sum(image * np.exp(-2j * np.pi * (u * np.arange(rows) / rows + v * np.arange(cols) / cols)))
        
            return result
        
        def fftshift(fft_result):
            # 将零频率分量移动到中心
            rows, cols = fft_result.shape
            # 计算频率分量的移位，以便零频率位于中心
            shift_rows = np.fft.ifftshift(np.arange(-rows // 2, rows // 2))
            shift_cols = np.fft.ifftshift(np.arange(-cols // 2, cols // 2))
        
            shifted_result = np.zeros_like(fft_result, dtype=np.complex128)
        
            for u in range(rows):
                for v in range(cols):
                    # 根据移位计算新的频域位置
                    shifted_result[u, v] = fft_result[(u + shift_rows) % rows, (v + shift_cols) % cols]
        
            return shifted_result          


        def fft_transform(image):
            # 应用2D傅里叶变换
            fft_result = np.fft.fft2(image)
            
            # 将零频率分量移动到中心
            fft_shifted = np.fft.fftshift(fft_result)
            
            # 计算幅度谱（对数尺度以获得更好的可视化效果）
            magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
            
            return fft_shifted, magnitude_spectrum
        

          
        def low_pass_filter(fft_shifted, cutoff_radius):
            # 获取图像的尺寸
            rows, cols = fft_shifted.shape
            
            # 创建低通滤波器的掩模
            mask = np.zeros_like(fft_shifted)
            center_row, center_col = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                    if distance <= cutoff_radius:
                        mask[i, j] = 1
            
            # 将掩模应用于傅里叶变换后的图像
            fft_shifted_low_pass = fft_shifted * mask
            
            return fft_shifted_low_pass
        
        def ifft_transform(fft_shifted):
            # 将频率分量移回原始位置
            fft_result = np.fft.ifftshift(fft_shifted)
            
            # 应用2D逆傅里叶变换
            image_reconstructed = np.fft.ifft2(fft_result).real
            
            return image_reconstructed

        
        def visualize_low_pass(image, magnitude_spectrum, reconstructed_image, reconstructed_spectrum):
            plt.figure(figsize=(16, 8))
            
            
            image = zoom(image, 4, order=1)
            reconstructed_image = zoom(reconstructed_image, 4, order=1)            
            magnitude_spectrum = zoom(magnitude_spectrum, 4, order=1)            
            reconstructed_spectrum = zoom(reconstructed_spectrum, 4, order=1)          
 
            plt.subplot(2, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
               
            plt.subplot(2, 3, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image (Low-pass Filtered)')
        

            plt.subplot(2, 3, 3)
            plt.imshow(np.abs(image - reconstructed_image), cmap='gray')
            plt.title('Difference Image')
            

            plt.subplot(2, 3, 4)
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum (Before)')
        

            plt.subplot(2, 3, 6)
            plt.imshow(np.abs(magnitude_spectrum - reconstructed_spectrum), cmap='gray')
            plt.title('Magnitude Spectrum Difference')

            plt.subplot(2, 3, 5)
            plt.imshow(reconstructed_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum (After)')
        
            plt.show()
            
        def high_pass_filter(fft_shifted, cutoff_radius):
            # 获取图像的尺寸
            rows, cols = fft_shifted.shape
            
            # 创建高通滤波器的掩模
            mask = np.ones_like(fft_shifted)
            center_row, center_col = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                    if distance <= cutoff_radius:
                        mask[i, j] = 0
            
            # 将掩模应用于傅里叶变换后的图像
            fft_shifted_high_pass = fft_shifted * mask
            
            return fft_shifted_high_pass

        
        
        def visualize_high_pass(image, magnitude_spectrum, reconstructed_image, reconstructed_spectrum):
            plt.figure(figsize=(16, 8))
            
            image = zoom(image, 4, order=1)
            reconstructed_image = zoom(reconstructed_image, 4, order=1)            
            magnitude_spectrum = zoom(magnitude_spectrum, 4, order=1)            
            reconstructed_spectrum = zoom(reconstructed_spectrum, 4, order=1)              
            
            plt.subplot(2, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
               
            plt.subplot(2, 3, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image (High-pass Filtered)')
        

            plt.subplot(2, 3, 3)
            plt.imshow(np.abs(image - reconstructed_image), cmap='gray')
            plt.title('Difference Image')
            

            plt.subplot(2, 3, 4)
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum (Before)')
        

            plt.subplot(2, 3, 6)
            plt.imshow(np.abs(magnitude_spectrum - reconstructed_spectrum), cmap='gray')
            plt.title('Magnitude Spectrum Difference')
        

            plt.subplot(2, 3, 5)
            plt.imshow(reconstructed_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum (After)')
        
            plt.show()
            
        def band_pass_filter(fft_shifted, low_cutoff_radius, high_cutoff_radius):
            # 获取图像的尺寸
            rows, cols = fft_shifted.shape
            
            # 创建带通滤波器的掩模
            mask = np.zeros_like(fft_shifted)
            center_row, center_col = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                    if low_cutoff_radius <= distance <= high_cutoff_radius:
                        mask[i, j] = 1
            
            # 将掩模应用于傅里叶变换后的图像
            fft_shifted_band_pass = fft_shifted * mask
            
            return fft_shifted_band_pass   
        
        def visualize_band_pass(image, magnitude_spectrum, reconstructed_image, reconstructed_spectrum):
            plt.figure(figsize=(16, 8))
            
                      
            image = zoom(image, 4, order=1)
            reconstructed_image = zoom(reconstructed_image, 4, order=1)            
            magnitude_spectrum = zoom(magnitude_spectrum, 4, order=1)            
            reconstructed_spectrum = zoom(reconstructed_spectrum, 4, order=1)            
         
            
            plt.subplot(2, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
               
            # 通道图像
            plt.subplot(2, 3, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image (Band-pass Filtered)')
        
            # 图像差异
            plt.subplot(2, 3, 3)
            plt.imshow(np.abs(image - reconstructed_image), cmap='gray')
            plt.title('Difference Image')
            
            # 幅度谱
            plt.subplot(2, 3, 4)
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum (Before)')
        
            # 幅度谱差异
            plt.subplot(2, 3, 6)
            plt.imshow(np.abs(magnitude_spectrum - reconstructed_spectrum), cmap='gray')
            plt.title('Magnitude Spectrum Difference')
        
            # 重构后的幅度谱
            plt.subplot(2, 3, 5)
            plt.imshow(reconstructed_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum (After)')
        
            plt.show()        
        
        def main():
            # 文件路径指向图像
            image_file = 'IMAGERY.TIF' 
            
            # 读取图像并将其转换为灰度图
            dataset = gdal.Open(image_file)
            #image_data = dataset.ReadAsArray()
            grayscale_image = dataset.GetRasterBand(1).ReadAsArray()
            
                                  
            # M/2 x N/2 的降采样图像
            grayscale_image = grayscale_image[::4, ::4]
            
            # 执行FFT
            fft_result, magnitude_spectrum = fft_transform(grayscale_image)

    

        
            print("############  Menu:  ############")
            print("1. 低通滤波")
            print("2. 高通滤波")
            print("3. 带通滤波")

        
            L2_option = int(input("Enter the L2_option (1/2/3): "))
            
            
            if L2_option == 1:
                # 定义低通滤波器的截止半径
                cutoff_radius = 0.03 * min(fft_result.shape)
                # 应用低通滤波器，去除了高频信息,平滑了图像,减少了图像细节
                fft_result_low_pass = low_pass_filter(fft_result, cutoff_radius)
                
                # 执行逆FFT以重建图像
                reconstructed_image_low_pass = ifft_transform(fft_result_low_pass)
                
                # 在低通滤波后计算幅度谱
                _, reconstructed_spectrum = fft_transform(reconstructed_image_low_pass)
                
                # 可视化图像和频谱图的差异
                visualize_low_pass(
                    grayscale_image, magnitude_spectrum,
                    reconstructed_image_low_pass, reconstructed_spectrum
                )
            
            if L2_option == 2:
                # 定义高通滤波器的截止半径
                cutoff_radius_high_pass = 0.03 * min(fft_result.shape)
                # 应用高通滤波器，实现了图像轮廓、边缘等高频信息的增强
                fft_result_high_pass = high_pass_filter(fft_result, cutoff_radius_high_pass)
                
                # 执行逆FFT以重建图像
                reconstructed_image_high_pass = ifft_transform(fft_result_high_pass)
                
                # 在高通滤波后计算幅度谱
                _, reconstructed_spectrum = fft_transform(reconstructed_image_high_pass)
                
                # 可视化图像和频谱图的差异
                visualize_high_pass(
                    grayscale_image, magnitude_spectrum,
                    reconstructed_image_high_pass, reconstructed_spectrum
                )
            
            if L2_option == 3:
                # 定义带通滤波器的截止半径
                low_cutoff_radius_band_pass = 0.01 * min(fft_result.shape)
                high_cutoff_radius_band_pass = 0.3 * min(fft_result.shape)
                
                # 应用带通滤波器，只保留图像傅里叶变换后频谱中在某个区间范围内的频率分量,滤除太高和太低的频率。


                fft_result_band_pass = band_pass_filter(fft_result, low_cutoff_radius_band_pass, high_cutoff_radius_band_pass)
                
                # 执行逆FFT以重建图像
                reconstructed_image_band_pass = ifft_transform(fft_result_band_pass)
                
                # 在带通滤波后计算幅度谱
                _, reconstructed_spectrum = fft_transform(reconstructed_image_band_pass)
                
                # 可视化图像和频谱图的差异
                visualize_band_pass(
                    grayscale_image, magnitude_spectrum,
                    reconstructed_image_band_pass, reconstructed_spectrum
                )

        
        if __name__ == "__main__":
           main()



    if L1_option == 3:

        
        def apply_pca(data, num_components):
            # 将数据重塑为2D数组（num_bands x num_pixels）
            bands, height, width = data.shape
            data_reshaped = data.reshape(bands, height * width).T
        
            # 标准化数据
            data_std = (data_reshaped - np.mean(data_reshaped, axis=0)) / np.std(data_reshaped, axis=0)
        
            # 计算协方差矩阵
            cov_matrix = np.cov(data_std, rowvar=False)
        
            # 执行特征值分解，得到特征值与特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
            # 将特征值和相应的特征向量按降序排序，[::-1]表示逆序,从大到小
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
        
            # 选择前k个特征向量形成变换矩阵（主成分矩阵）
            transformation_matrix = eigenvectors[:, :num_components]
        
            # 将标准化后的数据在主成分方向上的投影
            data_pca = np.dot(data_std, transformation_matrix)
        
            # 将PCA变换后的数据重塑回原始形状
            data_pca_reshaped = data_pca.T.reshape(num_components, height, width)
        
            return data_pca_reshaped
        
        def visualize_images(original, transformed):
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            #moveaxis适应matplotlib的channel_last图片绘制要求:频道维度(RGB通道数)需要在最后一轴。
            plt.imshow(np.moveaxis(original, 0, -1))
            plt.title('Original Image')
        

            plt.subplot(1, 2, 2)
            plt.imshow(np.moveaxis(transformed.astype('uint8'), 0, -1))
            plt.title('PCA Transformed Image')

        
            plt.show()
        
        if __name__ == "__main__":
            # 图像文件的路径
            image_file = 'IMAGERY.TIF'
        
            # 保留的主成分数量
            num_components = 3
        
            # 读取图像
            dataset = gdal.Open(image_file)
            image_data = dataset.ReadAsArray()
        
            # 应用PCA
            pca_transformed = apply_pca(image_data, num_components)
        
            # 可视化原始图像和PCA变换后的图像
            visualize_images(image_data, pca_transformed)
