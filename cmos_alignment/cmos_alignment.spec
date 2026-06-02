# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[('C:\\Users\\1\\miniconda3\\Library\\bin\\*.dll', '.')], # 注入底层 DLL 防闪退
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    exclude_binaries=False,     # 关键：设置为 False 才能生成真正的单文件
    name='CMOS校准系统',         # 这里修改你想要的软件名字
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,              # 设置为 False，隐藏黑色的 CMD 控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.ico'              # 如果有图标，确保把 OIP.ico 放在同级目录下；如果没有，可以删掉这行或换成你的图标名
)