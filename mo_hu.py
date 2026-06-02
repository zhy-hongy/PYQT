import os
from PIL import Image, ImageFilter

def find_first_image(folder="."):
    """返回文件夹中第一张图片的路径（jpg/png/jpeg）"""
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder, f)
    return None

def apply_gaussian_blur(input_path, output_dir, radii):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]

    for r in radii:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=r))
        out_path = os.path.join(output_dir, f"{base}_r{r}{ext}")
        blurred.save(out_path)
        print(f"已保存: {out_path}")

if __name__ == "__main__":
    # 自动查找当前目录下的第一张图片
    img_path = find_first_image()
    if img_path is None:
        print("当前目录下没有找到 jpg/png 图片。")
        img_path = input("请输入图片完整路径: ").strip().strip('"')
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            exit(1)

    print(f"处理图片: {img_path}")
    apply_gaussian_blur(img_path, "blurred_outputs", radii=[1, 3, 5, 10, 15])
    print("全部完成！")