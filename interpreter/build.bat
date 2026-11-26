@echo off
rem =============================================================================
rem build.bat - Builds the Portable C ISA Interpreter using Clang or GCC
rem =============================================================================
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

rem Try Clang from Xilinx tps first
if exist "C:\Xilinx\2025.1\tps\win64\clangd-8.0.0\bin\clang.exe" (
    echo [INFO] Compiling with Clang from Xilinx toolchain...
    if not exist "lib" mkdir "lib"
    "C:\Xilinx\2025.1\tps\win64\clangd-8.0.0\bin\clang.exe" -O2 -Wall -Iinclude -c src/mgc_interpreter.c -o src/mgc_interpreter.o
    "C:\Xilinx\2025.1\tps\win64\clangd-8.0.0\bin\clang.exe" -O2 -Wall -Iinclude -c src/mgc_fp16_math.c -o src/mgc_fp16_math.o
    "C:\Xilinx\2025.1\tps\win64\clangd-8.0.0\bin\llvm-ar.exe" rcs lib/libmgc_interpreter.a src/mgc_interpreter.o src/mgc_fp16_math.o
    "C:\Xilinx\2025.1\tps\win64\clangd-8.0.0\bin\clang.exe" -O2 -Wall -Iinclude test/test_interpreter.c lib/libmgc_interpreter.a -o test/test_interpreter.exe
    goto run
)

rem Try GCC from PATH
where gcc >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Compiling with GCC from PATH...
    gcc -O2 -Wall -Iinclude src/mgc_interpreter.c src/mgc_fp16_math.c test/test_interpreter.c -o test/test_interpreter.exe
    goto run
)

echo [ERROR] No suitable C compiler (clang or gcc) found!
exit /b 1

:run
if exist "test\test_interpreter.exe" (
    echo [SUCCESS] Compilation successful! Running test suite...
    echo ==================================================================
    test\test_interpreter.exe
    echo ==================================================================
)
