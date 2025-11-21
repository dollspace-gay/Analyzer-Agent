# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['F:\\ProtocolAI\\Protocol-AI\\ultimate_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('F:\\ProtocolAI\\Protocol-AI\\protocol_ai.py', '.'), ('F:\\ProtocolAI\\Protocol-AI\\modules', 'modules'), ('F:\\ProtocolAI\\Protocol-AI\\tools', 'tools'), ('F:\\ProtocolAI\\Protocol-AI\\gui', 'gui'), ('F:\\ProtocolAI\\Protocol-AI\\deep_research_agent.py', '.'), ('F:\\ProtocolAI\\Protocol-AI\\deep_research_integration.py', '.'), ('F:\\ProtocolAI\\Protocol-AI\\protocol_ai_logging.py', '.'), ('F:\\ProtocolAI\\Protocol-AI\\report_formatter.py', '.'), ('F:\\ProtocolAI\\Protocol-AI\\section_by_section_analysis.py', '.')],
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
