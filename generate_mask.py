import cv2
import numpy as np

def create_mask(output_path="mask.png"):
    """生成一个固定蒙版并保存为图像文件"""
    # 创建一个全黑的蒙版 (1090, 1920, 3)
    mask = np.zeros((1090, 1920, 3), dtype=np.uint8)

    # 设置正方形区域的边长和位置
    square_size = 300  # 正方形边长
    top_left = (810, 395)  # 左上角坐标 (x, y)
    bottom_right = (top_left[0] + square_size, top_left[1] + square_size)  # 右下角坐标

    # 在蒙版上绘制正方形（白色正方形）
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)  # 填充正方形

    # 保存蒙版为图像文件
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")

if __name__ == "__main__":
    # 调用函数生成蒙版
    create_mask(output_path="./static/mask.png")