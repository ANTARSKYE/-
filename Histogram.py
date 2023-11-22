# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:19:46 2023

@author: chen
"""
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import zoom

while True:
    print("############  Menu:  ############")
    print("1. 直方图")
    print("2. 线性拉伸")
    print("3. 二值化")
    print("4. 池化")
    print("0. Exit")

    L1_option = int(input("Enter the L1_option (0/1/2/3/4): "))

    if L1_option == 0:
        print("Exiting the program.")
        break  

    if L1_option == 1:
    
        ######################直方图调整功能#######################
        #直方图归一化
        def hist_normalization(image, a=0, b=255):
            c = image.min()  # 计算图像的最小像素值
            d = image.max()  # 计算图像的最大像素值
            out = image.copy()  # 复制输入图像
            out = (b - a) / (d - c) * (out - c) + a  # 直方图归一化公式
            out[out < a] = a  # 将小于a的像素值设置为a
            out[out > b] = b  # 将大于b的像素值设置为b
            out = out.astype(np.uint8)  # 将结果转换为8位无符号整数
            return out
        
        #直方图均衡化
        def hist_equalization(img):
            H, W = img.shape  # 获取图像的高度和宽度
            hist = np.histogram(img, bins=256, range=(0, 256))  # 计算图像的直方图
            cdf = hist[0].cumsum()  # 计算累积分布函数
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # 归一化累积分布函数
            out = cdf[img.astype(np.uint8)]  # 使用累积分布函数进行直方图均衡化
            hist_equalized = np.histogram(out, bins=256, range=(0, 256))  # 计算均衡化后图像的直方图
            return out, hist[0], hist_equalized[0]
        
        #直方图匹配
        def hist_matching(source, target):
            source_hist = np.histogram(source, bins=256, range=(0, 256))[0]  # 计算原图像的直方图
            target_hist = np.histogram(target, bins=256, range=(0, 256))[0]  # 计算目标图像的直方图
            source_cdf = source_hist.cumsum() / source_hist.sum()  # 计算原图像的累积分布函数
            target_cdf = target_hist.cumsum() / target_hist.sum()  # 计算目标图像的累积分布函数
            mapping_table = np.zeros(256, dtype=np.uint8)  # 创建映射表
            for i in range(256):
                diff = abs(source_cdf[i] - target_cdf)  # 计算累积分布函数之间的差异
                mapping_table[i] = np.argmin(diff)  # 找到最小差异的索引
            matched_image = mapping_table[source]  # 使用映射表进行直方图匹配
            return matched_image
                
        # 读取图像数据
        source_ds = gdal.Open("IMAGERY.TIF")
        target_ds = gdal.Open("IMAGERY2.TIF")
        source_band = source_ds.GetRasterBand(1)
        source_image = source_ds.GetRasterBand(1).ReadAsArray()
        target_image = target_ds.GetRasterBand(1).ReadAsArray()

        print("############  Menu:  ############")
        print("1. 直方图归一化")
        print("2. 直方图均衡化")
        print("3. 直方图匹配")
        
        option = int(input("Enter the option (1/2/3): "))
        
        if option == 1:
            #直方图标准化
            normalized_image = hist_normalization(source_image)
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(source_image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(source_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.subplot(2, 2, 3)
            plt.title("Normalized Image")
            plt.imshow(normalized_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("Normalized Histogram")
            plt.hist(normalized_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
        
        elif option == 2:
            # 直方图均衡化
            equalized_image, hist_original, hist_equalized = hist_equalization(source_image)
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(source_image, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("Original Histogram")
            plt.hist(source_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.subplot(2, 2, 3)
            plt.title("Equalized Image")
            plt.imshow(equalized_image, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("Equalized Histogram")
            plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()
        
        elif option == 3:
            # 直方图匹配
            matched_image = hist_matching(source_image, target_image)
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 2, 1)
            plt.title("Source Image")
            plt.imshow(source_image, cmap='gray')
            plt.subplot(3, 2, 2)
            plt.title("Source Histogram")
            plt.hist(source_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.subplot(3, 2, 3)
            plt.title("Target Image")
            plt.imshow(target_image, cmap='gray')
            plt.subplot(3, 2, 4)
            plt.title("Target Histogram")
            plt.hist(target_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.subplot(3, 2, 5)
            plt.title("Matched Image")
            plt.imshow(matched_image, cmap='gray')
            plt.subplot(3, 2, 6)
            plt.title("Matched Histogram")
            plt.hist(matched_image.ravel(), bins=256, range=(0, 256), density=True)
            plt.tight_layout()
            plt.show()

    if L1_option == 2:
        ######################线性拉伸功能#######################
        
        #0-255全局线性拉伸
        def linear_stretching(image):
            x_min = image.min()  # 计算图像的最小像素值
            x_max = image.max()  # 计算图像的最大像素值
            
            if x_min == x_max:
                return image
            
            stretched_image = ((255 - 0) / (x_max - x_min)) * (image - x_min) + 0  # 进行线性拉伸

            stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)  # 将结果限制在0到255之间并转换为8位无符号整数
            
            return stretched_image
        
        def two_percent_linear_stretching(image):
            #去掉2%百分位以下的数，去掉98%百分位以上的数
            lower_limit = np.percentile(image, 2)  # 计算2%分位数
            upper_limit = np.percentile(image, 98)  # 计算98%分位数
            
            stretched_image = ((255 - 0) / (upper_limit - lower_limit)) * (image - lower_limit) + 0  # 进行2%线性拉伸
            stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)  # 将结果限制在0到255之间并转换为8位无符号整数
            
            return stretched_image

        # 读取图像
        source_ds = gdal.Open("IMAGERY.TIF")
        image = source_ds.GetRasterBand(1).ReadAsArray()

        
        print("############  Menu:  ############")
        print("1. 线性拉伸")
        print("2. 2%线性拉伸")
        
        option = int(input("Enter the option (1/2): "))
        
        if option == 1:
            # 线性拉伸
            stretched_image = linear_stretching(image)
            plt.figure(figsize=(12, 9))
            plt.subplot(3, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(3, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            plt.subplot(3, 2, 3)
            plt.title("Linear Stretched Image")
            plt.imshow(stretched_image, cmap='gray')
            plt.subplot(3, 2, 4)
            plt.title("Linear Stretched Histogram")
            plt.hist(stretched_image.ravel(), bins=256, range=(0, 256), density=True)
        
        elif option == 2:
            # 2% 线性拉伸
            stretched_image_2percent = two_percent_linear_stretching(image)
            plt.figure(figsize=(12, 9))
            plt.subplot(3, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.subplot(3, 2, 2)
            plt.title("Original Histogram")
            plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
            plt.subplot(3, 2, 3)
            plt.title("2% Linear Stretched Image")
            plt.imshow(stretched_image_2percent, cmap='gray')
            plt.subplot(3, 2, 4)
            plt.title("2% Linear Stretched Histogram")
            plt.hist(stretched_image_2percent.ravel(), bins=256, range=(0, 256), density=True)
        
        plt.tight_layout()
        plt.show()

    if L1_option == 3:
        ######################二值化功能#######################
        
        def threshold_binary(image, threshold=128):
            binary_image = np.zeros_like(image)  # 创建与输入图像相同大小的零数组
            binary_image[image >= threshold] = 255  # 使用阈值进行二值化
            return binary_image
        
        def otsu_threshold(image):
            best_threshold = 0  # 初始化最佳阈值
            max_variance = 0  # 初始化最大方差
            h, w = image.shape  # 获取图像的高度和宽度
            hist = np.histogram(image, bins=256, range=(0, 256))[0]  # 计算图像的直方图
            total_pixels = h * w  # 计算总像素数
        
            def calculate_intra_class_variance(hist, t):
                w0 = hist[:t].sum() / total_pixels  # 计算第一个类别的权重
                w1 = hist[t:].sum() / total_pixels  # 计算第二个类别的权重
                if w0 == 0 or w1 == 0:
                    return 0
                m0 = np.dot(np.arange(t), hist[:t]).sum() / (w0 * total_pixels)  # 计算第一个类别的平均灰度
                m1 = np.dot(np.arange(t, 256), hist[t:]).sum() / (w1 * total_pixels)  # 计算第二个类别的平均灰度
                return w0 * w1 * (m0 - m1) ** 2  # 计算类间方差
        
            for t in range(1, 255):
                variance = calculate_intra_class_variance(hist, t)  # 计算特定阈值下的方差
                if variance > max_variance:
                    max_variance = variance
                    best_threshold = t  # 更新最佳阈值
        
            binary_image = np.where(image > best_threshold, 255, 0).astype(np.uint8)  # 使用最佳阈值进行二值化
        
            return binary_image, best_threshold

        # 读取图像
        source_ds = gdal.Open("IMAGERY.TIF")
        image = source_ds.GetRasterBand(1).ReadAsArray()
        
        print("############  Menu:  ############")
        print("1. 阈值二值化")
        print("2. 大津二值化")
        
        option = int(input("Enter the option (1/2): "))
        
        if option == 1:
            threshold = int(input("Enter the threshold value (0-255): "))
            binary_image = threshold_binary(image, threshold)
           
            title = f"Threshold Binary (Threshold {threshold})"
        elif option == 2:
            binary_image, best_threshold = otsu_threshold(image)
            
            title = f"Otsu Binary (Threshold {best_threshold})"
        
        # 绘制原始图像和二值化后的图像
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.title("Original Histogram")
        plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
        
        plt.subplot(2, 2, 3)
        plt.title(title)
        plt.imshow(binary_image, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.title(title + " Histogram")
        plt.hist(binary_image.ravel(), bins=256, range=(0, 256), density=True)
        
        plt.tight_layout()
        plt.show()
        

        
        

    elif  L1_option == 4:
        ######################池化功能#######################
        
        # 定义平均池化函数
        def average_pooling(image, grid_size):
            h, w = image.shape  # 获取图像的高度和宽度
            gh, gw = grid_size  # 获取网格的行数和列数
            cell_height = h // gh  # 计算每个单元格的高度
            cell_width = w // gw  # 计算每个单元格的宽度
            pooled_image = np.zeros((h, w), dtype=np.uint8)  # 创建与输入图像相同大小的零数组
            for i in range(gh):
                for j in range(gw):
                    start_h = i * cell_height  # 计算单元格的起始行
                    end_h = (i + 1) * cell_height  # 计算单元格的结束行
                    start_w = j * cell_width  # 计算单元格的起始列
                    end_w = (j + 1) * cell_width  # 计算单元格的结束列
                    average_value = np.mean(image[start_h:end_h, start_w:end_w])  # 计算单元格内像素的平均值
                    pooled_image[start_h:end_h, start_w:end_w] = average_value  # 将平均值赋值给池化后的图像
            return pooled_image
        
        # 定义最大池化函数
        def max_pooling(image, grid_size):
            h, w = image.shape  # 获取图像的高度和宽度
            gh, gw = grid_size  # 获取网格的行数和列数
            cell_height = h // gh  # 计算每个单元格的高度
            cell_width = w // gw  # 计算每个单元格的宽度
            pooled_image = np.zeros((h, w), dtype=np.uint8)  # 创建与输入图像相同大小的零数组
            for i in range(gh):
                for j in range(gw):
                    start_h = i * cell_height  # 计算单元格的起始行
                    end_h = (i + 1) * cell_height  # 计算单元格的结束行
                    start_w = j * cell_width  # 计算单元格的起始列
                    end_w = (j + 1) * cell_width  # 计算单元格的结束列
                    max_value = np.max(image[start_h:end_h, start_w:end_w])  # 计算单元格内像素的最大值
                    pooled_image[start_h:end_h, start_w:end_w] = max_value  # 将最大值赋值给池化后的图像
            return pooled_image

        
        # 读取图像
        source_ds = gdal.Open("IMAGERY.TIF")
        image = source_ds.GetRasterBand(1).ReadAsArray()
        # 定义池化的网格大小
        grid_size = (20, 20)
        
        print("############  Menu:  ############")
        print("1. 平均池化")
        print("2. 最大池化")
        
        option = int(input("Enter the option (1/2): "))
        
        if option == 1:
            # 应用平均池化操作
            pooled_image = average_pooling(image, grid_size)
            title = "Average Pooling"
        elif option == 2:
            # 应用最大池化操作
            pooled_image = max_pooling(image, grid_size)
            title = "Max Pooling"
        
        # 绘制原始图像和池化后的图像
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.title("Original Histogram")
        plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
        
        plt.subplot(2, 2, 3)
        plt.title(title)
        plt.imshow(pooled_image, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.title(title + " Histogram")
        plt.hist(pooled_image.ravel(), bins=256, range=(0, 256), density=True)
        
        plt.tight_layout()
        plt.show()

