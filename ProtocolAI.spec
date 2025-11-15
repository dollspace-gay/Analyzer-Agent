# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['f:\\Agent\\ultimate_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('f:\\Agent\\protocol_ai.py', '.'), ('f:\\Agent\\modules', 'modules'), ('f:\\Agent\\tools', 'tools'), ('f:\\Agent\\gui', 'gui'), ('f:\\Agent\\deep_research_agent.py', '.'), ('f:\\Agent\\deep_research_integration.py', '.'), ('f:\\Agent\\protocol_ai_logging.py', '.'), ('f:\\Agent\\report_formatter.py', '.'), ('f:\\Agent\\section_by_section_analysis.py', '.')],
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
    a.datas,
    [],
    name='ProtocolAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
