@echo on
REM
setlocal
cd /D "%~dp0"
call %UserProfile%\anaconda3\Scripts\activate.bat
REM
call conda activate ai6121cv
REM
call jupyter lab