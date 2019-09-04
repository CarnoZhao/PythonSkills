using DelimitedFiles, Statistics, Random
using Flux
using CUDAdrv, CuArrays, CUDAnative

function load_data()
    data = readdlm("/home/tongxueqing/zhaox/codes/PythonSkills/YahooStock/sp500_total_close.csv", ',',  header = true, )[1]
    size(data)
end