using Distributions, Statistics, LaTeXStrings, LinearAlgebra, Plots, DataFrames

include("02probestim.jl")

# Datos observados
r = 23
n = 100
θ = 0.9
x_obs = rand(Binomial(r, θ), n)

function estadisticos_sumario(data) 
    return sort(data)
end

# vector de estadisticos de orden entre el obs contra el simulado 

s_obs=estadisticos_sumario(x_obs)

using Distributions, Statistics

function abc_r_distancia_absoluta(x_obs, θ, N)
    
    n = length(x_obs)
    s_obs = estadisticos_sumario(x_obs)   
    
    r = Int[]
    d = Float64[]
    
    for i in 1:N
        # prior discreto uniforme
        r_sim = rand(maximum(x_obs):50)
        
        # simular datos
        x_sim = rand(Binomial(r_sim, θ), n)
        s_sim = estadisticos_sumario(x_sim)
        
        # guardar
        push!(r, r_sim)
        push!(d, sum(abs.(s_sim .- s_obs)))
    end
    
    return r, d

end

function abc_r_distancia_euclidiana(x_obs, θ, N)
    
    n = length(x_obs)
    
    r = Int[]
    d = Float64[]

    
    for i in 1:N
        r_sim = rand(maximum(x_obs):50)
        x_sim = rand(Binomial(r_sim, θ), n)
        s_sim = estadisticos_sumario(x_sim)

        push!(r, r_sim)
        push!(d, norm(s_sim .- s_obs))
    end
    
    return r, d
end

# =========================
# Intervalo creíble HPD
# =========================

function intervalo_creible(muestras, α=0.05)

    x = sort(muestras)

    n = length(x)
    m = Int(floor((1-α)*n))

    mejor_long = Inf
    li = 0
    ls = 0

    for i in 1:(n-m)

        a = x[i]
        b = x[i+m]

        if (b-a) < mejor_long
            mejor_long = b-a
            li = a
            ls = b
        end
    end

    return (li, ls)
end

function abc_promedio_r(x_obs, θ, N, m; k_frac = 0.001)

    r_grid = collect(maximum(x_obs):50)

    post_abs = zeros(length(r_grid))
    post_euc = zeros(length(r_grid))

    resumen = DataFrame(
        Metodo = String[],
        Corrida = Int[],
        Media = Float64[],
        Mediana = Float64[],
        Moda = Float64[],
        HPD_inf = Float64[],
        HPD_sup = Float64[]
    )

    for sim in 1:m

        println("Iteración ", sim)

        ######## ABC ABSOLUTA ########

        r_abs0, d_abs0 = abc_r_distancia_absoluta(x_obs, θ, N)

        k0 = Int(trunc(k_frac * length(d_abs0)))

        idx_abs0 = sortperm(d_abs0)[1:k0]

        r_abs_acc0 = r_abs0[idx_abs0]

        R_abs0 = masaprob(r_abs_acc0)

        probs_temp = zeros(length(r_grid))

        for i in eachindex(R_abs0.valores)

            pos = findfirst(==(R_abs0.valores[i]), r_grid)

            if pos !== nothing
                probs_temp[pos] = R_abs0.probs[i]
            end
        end

        post_abs .+= probs_temp

        IC_abs0 = intervalo_creible(r_abs_acc0)

        push!(
            resumen,
            (
                "ABC absoluta",
                sim,
                mean(r_abs_acc0),
                median(r_abs_acc0),
                mode(r_abs_acc0),
                IC_abs0[1],
                IC_abs0[2]
            )
        )

        ######## ABC EUCLIDIANA ########

        r_euc0, d_euc0 = abc_r_distancia_euclidiana(x_obs, θ, N)

        idx_euc0 = sortperm(d_euc0)[1:k0]

        r_euc_acc0 = r_euc0[idx_euc0]

        R_euc0 = masaprob(r_euc_acc0)

        probs_temp = zeros(length(r_grid))

        for i in eachindex(R_euc0.valores)

            pos = findfirst(==(R_euc0.valores[i]), r_grid)

            if pos !== nothing
                probs_temp[pos] = R_euc0.probs[i]
            end
        end

        post_euc .+= probs_temp

        IC_euc0 = intervalo_creible(r_euc_acc0)

        push!(
            resumen,
            (
                "ABC euclidiana",
                sim,
                mean(r_euc_acc0),
                median(r_euc_acc0),
                mode(r_euc_acc0),
                IC_euc0[1],
                IC_euc0[2]
            )
        )

    end

    post_abs ./= m
    post_euc ./= m

    metodos = unique(resumen.Metodo)

    promedios = DataFrame(
        Metodo = String[],
        Media_promedio = Float64[],
        Mediana_promedio = Float64[],
        Moda_promedio = Float64[],
        HPD_inf_promedio = Float64[],
        HPD_sup_promedio = Float64[]
    )

    for metodo in metodos

        df = resumen[resumen.Metodo .== metodo, :]

        push!(
            promedios,
            (
                metodo,
                mean(df.Media),
                median(df.Mediana),
                mode(df.Moda),
                mean(df.HPD_inf),
                mean(df.HPD_sup)
            )
        )

    end

    p = bar(r_grid ,post_abs,width = 0.4,alpha = 0.6,label = "ABC absoluta promedio",xlabel = "r",ylabel = "Probabilidad",title = L"Posterior\;promedio\;\hat{\rho}_m(r\mid x_{obs})=\frac{1}{m}\sum_{j=1}^m\rho_{ABC,j}(r\mid\mathbf{x}_{obs})")
    bar!(r_grid,post_euc,width = 0.4,alpha = 0.6,label = "ABC euclidiana promedio")
    vline!([r],label=L"Valor\;verdadero\;r",lw=3,color=:purple)
    display(p)

    println()
    println("===== PROMEDIO ESTIMACIONES =====")
    println(promedios)

    return (resumen = resumen,promedios = promedios,post_abs = post_abs,post_euc = post_euc,r_grid = r_grid,grafica = p)

end

N = 200000

r_abs, d_abs = abc_r_distancia_absoluta(x_obs, θ, N);
r_euc, d_euc = abc_r_distancia_euclidiana(x_obs, θ, N);

d_abs_sorted = sort(d_abs)
d_euc_sorted = sort(d_euc)
k = Int(0.0005 * length(d_abs))
ϵ_abs = d_abs_sorted[k]
ϵ_euc = d_euc_sorted[k]

# Crear gráfica
p = plot(sort(d_abs),label="Distancia absoluta",xlabel="Índice (ordenado)",ylabel="Distancia",title=L"Distancias\;ordenadas\;\epsilon_{(i)}",lw=2)
plot!(p, sort(d_euc),label="Distancia euclidiana",lw=2)

hline!(p, [ϵ_abs], label=L"\epsilon_{abs}")
hline!(p, [ϵ_euc], label=L"\epsilon_{euc}", linestyle=:dash)

vline!(p, [k], label="k-ésimo orden", linestyle=:dot)

display(p)

k = Int(trunc(0.001 * length(d_abs)))

idx_abs = sortperm(d_abs)[1:k]
r_abs_aceptados = r_abs[idx_abs]

idx_euc = sortperm(d_euc)[1:k]
r_euc_aceptados = r_euc[idx_euc]

R_abs = masaprob(r_abs_aceptados)
R_euc = masaprob(r_euc_aceptados)

p2 = bar(R_abs.valores, R_abs.probs,label="ABC absoluto",xlabel="r",ylabel=L"\rho(r\mid\mathbf{x}_{obs})",title=L"\rho_{ABC}(r\mid \mathbf{x}_{obs}),\ \theta = %$θ",alpha=0.6)
bar!(R_euc.valores, R_euc.probs,label="ABC euclidiano",alpha=0.6)

display(p2)

# =========================
# Intervalos HPD
# =========================

IC_abs = intervalo_creible(r_abs_aceptados, 0.05)
IC_euc = intervalo_creible(r_euc_aceptados, 0.05)

# =========================
# DataFrame resumen
# =========================

sumario = DataFrame(
    Metodo = ["ABC absoluta", "ABC euclidiana"],
    
    Media = [
        mean(r_abs_aceptados),
        mean(r_euc_aceptados)
    ],

    Mediana = [
        median(r_abs_aceptados),
        median(r_euc_aceptados)
    ],

    Moda = [
        mode(r_abs_aceptados),
        mode(r_euc_aceptados)
    ],

    HPD_95_inf = [
        IC_abs[1],
        IC_euc[1]
    ],

    HPD_95_sup = [
        IC_abs[2],
        IC_euc[2]
    ],

    Longitud = [
        IC_abs[2] - IC_abs[1],
        IC_euc[2] - IC_euc[1]
    ]
)

print(sumario)

N = 200000 #numero de simulaciones en ABC
m = 30 #numero de veces que se hace el abc para promediarlo
k_frac = 0.001

println()

resultado = abc_promedio_r(x_obs, θ, 200000, m);