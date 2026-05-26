
using Distributions, Statistics, LaTeXStrings, LinearAlgebra, Plots, DataFrames

include("02probestim.jl")

# Datos observados
r = 20
n = 50
θ = 0.1
x_obs = rand(Binomial(r, θ), n)

function estadisticos_sumario(data) 
    return sum(data)
end

function abc_distancia_absoluta(x_obs, r, N)
    
    n = length(x_obs)
    s_obs = estadisticos_sumario(x_obs)  # estadístico resumen
    
    θ = Float64[]
    d = Float64[]
    
    for i in 1:N
        θ_sim = rand(Uniform(0,1))
        x_sim = rand(Binomial(r, θ_sim), n)
        s_sim = estadisticos_sumario(x_sim)
        
        push!(θ, θ_sim)
        push!(d, abs(s_sim - s_obs))
    end
    
    return θ, d
end


function abc_distancia_euclidiana(x_obs, r, N)
    
    n = length(x_obs)
    
    θ = Float64[]
    d = Float64[]
    
    for i in 1:N
        θ_sim = rand(Uniform(0,1))
        x_sim = rand(Binomial(r, θ_sim), n)
        
        push!(θ, θ_sim)
        push!(d, norm(x_sim - x_obs))
    end
    
    return θ, d
end


function intervalo_longitud_minima(D, α=0.95)
    x = D.reps
    f = D.fdp.(x)
    h = (D.max - D.min) / D.nclases
    p = f .* h
    n = length(x)
    min_len = Inf
    best_interval = (0.0, 0.0)
    for i in 1:n
        suma = 0.0
        for j in i:n
            suma += p[j]
            if suma ≥ α
                len = x[j] - x[i]
                if len < min_len
                    min_len = len
                    best_interval = (x[i], x[j])
                end
                break
            end
        end
    end
    return best_interval, min_len
end

function intervalo_longitud_minima_beta(θ_post, α=0.95; grid_size=10_000)
    
    x = range(0, 1, length=grid_size)
    
    f = pdf.(θ_post, x)
    
    h = step(x)
    p = f .* h
    
    n = length(x)
    min_len = Inf
    best_interval = (0.0, 0.0)
    
    for i in 1:n
        suma = 0.0
        for j in i:n
            suma += p[j]
            if suma ≥ α
                len = x[j] - x[i]
                if len < min_len
                    min_len = len
                    best_interval = (x[i], x[j])
                end
                break
            end
        end
    end
    
    return best_interval, min_len
end

N = 100_000

θ_abs, d_abs = abc_distancia_absoluta(x_obs, r, N)
θ_euc, d_euc = abc_distancia_euclidiana(x_obs, r, N)

p1=plot(sort(d_abs), label="Distancia absoluta", xlabel="Índice (ordenado)", ylabel="Distancia",title=L"Distancias\; ordenadas\;\epsilon_{(k)} ",lw=2)
plot!(sort(d_euc), label="Distancia euclidiana", lw=2)
hline!([sort(d_abs)[Int(trunc(0.0005*length(d_abs)))]], label = L"\epsilon_{(k)}",lw=2)

display(p1)

idx_abs_sorted = sortperm(d_abs)
k_abs = Int(0.0015 * length(d_abs)) #elegimos el valor de k
idx_abs_aceptados = idx_abs_sorted[1:k_abs]
θ_abs_aceptados = θ_abs[idx_abs_aceptados]

idx_euc_sorted = sortperm(d_euc)
k_euc = Int(trunc(0.0015 * length(d_euc)))
idx_euc_aceptados = idx_euc_sorted[1:k_euc]
θ_euc_aceptados = θ_euc[idx_euc_aceptados]

# Estimaciones no parámetricas de la densidad 

D_abs = densprob(θ_abs_aceptados)
D_euc = densprob(θ_euc_aceptados)

# Familia conjugada 

θ_post=Beta(sum(x_obs)+1,length(x_obs)*r-sum(x_obs)+1)
θ_grid = range(0, 1, length=1000)

p2=plot(D_abs.reps, D_abs.fdp.(D_abs.reps),label=L"\pi(\theta|\textbf{x}_{obs})\; absoluta",xlabel="θ",ylabel=L"\pi_{ABC}(\theta\mid\mathbf{x}_{obs})",title=L"\pi_{ABC}(\theta\mid\mathbf{x}_{obs}),(\theta = %$θ)",lw=2)
plot!(D_euc.reps, D_euc.fdp.(D_euc.reps),label=L"\pi(\theta|\textbf{x}_{obs})\; euclidiana",lw=2)
plot!(θ_grid, pdf.(θ_post, θ_grid),label=L"\pi(\theta|\textbf{x}_{obs})\; teórica",lw=2)
display(p2)

intervalo_abs, long_abs = intervalo_longitud_minima(D_abs, 0.95)
intervalo_euc, long_euc = intervalo_longitud_minima(D_euc, 0.95)
intervalo_beta, long_beta = intervalo_longitud_minima_beta(θ_post, 0.95)

println("=== Distancia absoluta ===")
println("Intervalo: ", intervalo_abs)
println("Longitud mínima: ", long_abs)

println("\n=== Distancia euclidiana ===")
println("Intervalo: ", intervalo_euc)
println("Longitud mínima: ", long_euc)

println("=== Posterior Beta ===")
println("Intervalo HDI: ", intervalo_beta)
println("Longitud mínima: ", long_beta)


tabla_estim = DataFrame(
    Metodo = ["ABC abs", "ABC euc", "Beta"],
    
    Limite_inferior = [intervalo_abs[1], intervalo_euc[1], intervalo_beta[1]],
    Limite_superior = [intervalo_abs[2], intervalo_euc[2], intervalo_beta[2]],
    Longitud = [long_abs, long_euc, long_beta],
    
    Media = [
        mean(θ_abs_aceptados),
        mean(θ_euc_aceptados),
        mean(θ_post)
    ],
    
    Mediana = [
        median(θ_abs_aceptados),
        median(θ_euc_aceptados),
        quantile(θ_post, 0.5)
    ]
)
println(tabla_estim)


p3=plot(xlabel="θ", ylabel="",title="Intervalos de longitud mínima",yticks=([1,2,3], ["ABC abs", "ABC euc", "Beta"]),legend=:topright)

# Intervalos creibles 95% de longitud mínima

plot!([intervalo_abs[1], intervalo_abs[2]], [1,1],lw=4, label="ABC abs")
plot!([intervalo_euc[1], intervalo_euc[2]], [2,2],lw=4, label="ABC euc")
plot!([intervalo_beta[1], intervalo_beta[2]], [3,3],lw=4, label="Beta")

# estimaciones puntuales bajo perdida cuadrática

scatter!([mean(θ_abs_aceptados)], [1],label=L"\hat{\theta}_{ABC}")
scatter!([mean(θ_euc_aceptados)], [2],label=L"\hat{\theta}_{EUC}")
scatter!([mean(θ_post)], [3],label=L"\hat{\theta}_{conj}")

display(p3)

# Encontrar el intervalo más largo
intervalos = [intervalo_abs, intervalo_euc, intervalo_beta]
longitudes = [long_abs, long_euc, long_beta]

idx_max = argmax(longitudes)
intervalo_mas_largo = intervalos[idx_max]

# Límites para hacer zoom
xmin = max(0, intervalo_mas_largo[1] - 0.05)
xmax = min(1, intervalo_mas_largo[2] + 0.05)

# Malla fina para suavizar
θ_suave = range(xmin, xmax, length=5000)

# Evaluar densidades en la malla fina
dens_abs = D_abs.fdp.(θ_suave)
dens_euc = D_euc.fdp.(θ_suave)
dens_beta = pdf.(θ_post, θ_suave)

p4 = plot(θ_suave,dens_abs,label = L"\pi(\theta|\mathbf{x}_{obs})\; absoluta",xlabel = "θ",ylabel = L"\pi(\theta|\mathbf{x}_{obs})",title = L"\pi_{ABC}(\theta|\mathbf{x}_{obs}),(\theta = %$θ)",lw = 3,xlims = (xmin, xmax))
plot!(θ_suave,dens_euc,label = L"\pi(\theta|\mathbf{x}_{obs})\; euclidiana",lw = 3)
plot!(θ_suave,dens_beta,label = L"\pi(\theta|\mathbf{x}_{obs})\; teórica",lw = 3) 
display(p4)