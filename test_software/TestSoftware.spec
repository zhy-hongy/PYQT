# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=['app'],
    binaries=[('C:\\Users\\1\\miniconda3\\Library\\bin\\*.dll', '.')],
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
    a.binaries,  # <--- 把这两行搬到这里来
    a.zipfiles,  # <--- 把这两行搬到这里来
    a.datas,     # <--- 把这两行搬到这里来
    exclude_binaries=False, # 1. 必须改成 False！允许把二进制文件打包进单个 exe
    name='激光线校对',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,upx_exclude=[],          # 2. 补上这个单文件必需的空列表参数
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='OIP.ico'  # <--- 在这里添加或修改图标路径（确保logo.ico在项目根目录下）
)
#coll = COLLECT(
#    exe,
#    a.binaries,
#    a.datas,
#    strip=False,
#    upx=True,
#    upx_exclude=[],
#    name='TestSoftware',
#)
