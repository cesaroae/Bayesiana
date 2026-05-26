
using Distributions, Statistics, Random, LinearAlgebra, Plots, LaTeXStrings, QuadGK

include("02probestim.jl")
include("03discreta.jl")

Random.seed!(123)

n_true = 20

Θ_true = 0.1

N = 100

data_obs = rand(Binomial(n_true, Θ_true), N)

# =====================================================
#        Cálculo Bayesiano Aproximado
# =====================================================

function estadisticos_sumario(data) # función que calcula los summary statistics
    return [mean(data), sqrt(var(data))]
end

function estadisticos_momentos(data)
    m = mean(data)
    v = var(data)
    if m == 0
        return [0.0, 0.0]
    end
    θ_hat = 1 - v/m
    # estabilidad numérica
    θ_hat = clamp(θ_hat, 1e-8, 1 - 1e-8)
    n_hat = m / θ_hat
    return [n_hat, θ_hat]
end

function estadisticos_robustos(data)

    med = median(data)
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)

    iqr = q3 - q1

    return [med, iqr]
end

function estadisticos_orden(data)

    x_sorted = sort(data)

    x1 = x_sorted[1]

    q25 = quantile(x_sorted, 0.25)

    med = quantile(x_sorted, 0.50)

    q75 = quantile(x_sorted, 0.75)

    xn = x_sorted[end]

    return [x1, q25, med, q75, xn]
end


function distancia_euclidiana(s1, s2)
    return norm(s1 - s2)
end

function distancia_absoluta(s1, s2)
    return sum(abs.(s1 .- s2))
end

function sample_prior()
    n = rand(maximum(data_obs):100) # uniforme sobre {1,...,100}
    Θ = rand(Beta(1,1))# beta(1,1)=Uniforme(0,1)
    return n, Θ
end

# ================================
# Cálculo Aproximado Bayesiano
# ================================

function abc_completo(N_sim)

    n_total = Int[]
    Θ_total = Float64[]
    s_obs=estadisticos_sumario(data_obs)
    dist_eucl = Float64[]
    dist_abs = Float64[]

    for i in 1:N_sim

        n, Θ = sample_prior()

        push!(n_total, n)
        push!(Θ_total, Θ)

        sim_data = rand(Binomial(n, Θ), N)

        s_sim = estadisticos_sumario(sim_data)

        d1 = distancia_euclidiana(s_sim, s_obs)
        d2 = distancia_absoluta(s_sim, s_obs)

        push!(dist_eucl, d1)
        push!(dist_abs, d2)

    end

    return n_total, Θ_total, dist_eucl, dist_abs
end

# =====================================================
# ABC MOMENTOS
# =====================================================

function abc_momentos(N_sim)

    n_total = Int[]
    Θ_total = Float64[]
    s_obs_mom = estadisticos_momentos(data_obs)
    dist_eucl = Float64[]
    dist_abs = Float64[]

    for i in 1:N_sim

        n, Θ = sample_prior()

        push!(n_total, n)
        push!(Θ_total, Θ)

        sim_data = rand(Binomial(n, Θ), N)

        s_sim=estadisticos_momentos(sim_data)

        d1 = distancia_euclidiana(s_sim, s_obs_mom)
        d2 = distancia_absoluta(s_sim, s_obs_mom)

        push!(dist_eucl, d1)
        push!(dist_abs, d2)

    end

    return n_total, Θ_total, dist_eucl, dist_abs
end

# =====================================================
# ABC ROBUSTO
# =====================================================

function abc_robusto(N_sim)

    n_total = Int[]
    Θ_total = Float64[]
    s_obs_rob=estadisticos_robustos(data_obs)
    dist_eucl = Float64[]
    dist_abs = Float64[]

    for i in 1:N_sim

        n, Θ = sample_prior()

        push!(n_total, n)
        push!(Θ_total, Θ)

        sim_data = rand(Binomial(n, Θ), N)

        s_sim = estadisticos_robustos(sim_data)

        d1 = distancia_euclidiana(s_sim, s_obs_rob)
        d2 = distancia_absoluta(s_sim, s_obs_rob)

        push!(dist_eucl, d1)
        push!(dist_abs, d2)

    end

    return n_total, Θ_total, dist_eucl, dist_abs
end

# =====================================================
# ABC ORDEN 
# =====================================================

function abc_orden(N_sim)

    n_total = Int[]
    Θ_total = Float64[]
    s_obs_ord=estadisticos_orden(data_obs)
    dist_eucl = Float64[]
    dist_abs = Float64[]

    for i in 1:N_sim

        n, Θ = sample_prior()

        push!(n_total, n)
        push!(Θ_total, Θ)

        sim_data = rand(Binomial(n, Θ), N)

        s_sim = estadisticos_orden(sim_data)

        d1 = distancia_euclidiana(s_sim, s_obs_ord)
        d2 = distancia_absoluta(s_sim, s_obs_ord)

        push!(dist_eucl, d1)
        push!(dist_abs, d2)

    end

    return n_total, Θ_total, dist_eucl, dist_abs
end

N_sim = 100000

# Original (media muestral, desviacion estandar)
n_abc1, θ_abc1, diste1, dista1 = abc_completo(N_sim);

# Momentos (estimadores de momentos)
n_abc2, θ_abc2, diste2, dista2 = abc_momentos(N_sim);

# Robusto (mediana y rango intercuantilico)
n_abc3, θ_abc3, diste3, dista3 = abc_robusto(N_sim);

# Orden (Cuantiles, minimo y maximo muestral)
n_abc4, θ_abc4, diste4, dista4 = abc_orden(N_sim);

# =====================================================
# RECHAZO ABC
# =====================================================

k_euc = Int(floor(0.003 * length(dista1)))

k_abs = Int(floor(0.003 * length(dista1)))

# =====================================================
# ABC 1
# =====================================================

idx_euc1_sorted = sortperm(diste1)

idx_euc1_aceptados = idx_euc1_sorted[1:k_euc]

n_euc1_aceptados = n_abc1[idx_euc1_aceptados]
θ_euc1_aceptados = θ_abc1[idx_euc1_aceptados]

idx_abs1_sorted = sortperm(dista1)

idx_abs1_aceptados = idx_abs1_sorted[1:k_abs]

n_abs1_aceptados = n_abc1[idx_abs1_aceptados]
θ_abs1_aceptados = θ_abc1[idx_abs1_aceptados]


# =====================================================
# ABC MOMENTOS 
# =====================================================

idx_euc2_sorted = sortperm(diste2)

idx_euc2_aceptados = idx_euc2_sorted[1:k_euc]

n_euc2_aceptados = n_abc2[idx_euc2_aceptados]
θ_euc2_aceptados = θ_abc2[idx_euc2_aceptados]

idx_abs2_sorted = sortperm(dista2)

idx_abs2_aceptados = idx_abs2_sorted[1:k_abs]

n_abs2_aceptados = n_abc2[idx_abs2_aceptados]
θ_abs2_aceptados = θ_abc2[idx_abs2_aceptados]


# =====================================================
# ABC 3 MEDIANA Y RIQ
# =====================================================

idx_euc3_sorted = sortperm(diste3)

idx_euc3_aceptados = idx_euc3_sorted[1:k_euc]

n_euc3_aceptados = n_abc3[idx_euc3_aceptados]
θ_euc3_aceptados = θ_abc3[idx_euc3_aceptados]

idx_abs3_sorted = sortperm(dista3)

idx_abs3_aceptados = idx_abs3_sorted[1:k_abs]

n_abs3_aceptados = n_abc3[idx_abs3_aceptados]
θ_abs3_aceptados = θ_abc3[idx_abs3_aceptados]


# =====================================================
# ABC 4 
# =====================================================

idx_euc4_sorted = sortperm(diste4)

idx_euc4_aceptados = idx_euc4_sorted[1:k_euc]

n_euc4_aceptados = n_abc4[idx_euc4_aceptados]
θ_euc4_aceptados = θ_abc4[idx_euc4_aceptados]

idx_abs4_sorted = sortperm(dista4)

idx_abs4_aceptados = idx_abs4_sorted[1:k_abs]

n_abs4_aceptados = n_abc4[idx_abs4_aceptados]
θ_abs4_aceptados = θ_abc4[idx_abs4_aceptados]

# =====================================================
# DISTANCIAS TOTALES ORDENADAS
# =====================================================

diste1_ord = sort(diste1)
dista1_ord = sort(dista1)

diste2_ord = sort(diste2)
dista2_ord = sort(dista2)

diste3_ord = sort(diste3)
dista3_ord = sort(dista3)

diste4_ord = sort(diste4)
dista4_ord = sort(dista4)


# =====================================================
# ABC 1
# =====================================================

p1 = plot(
    diste1_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Original"
)

plot!(
    dista1_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# ABC 2
# =====================================================

p2 = plot(
    diste2_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Momentos"
)

plot!(
    dista2_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# ABC 3
# =====================================================

p3 = plot(
    diste3_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Robusto"
)

plot!(
    dista3_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# ABC 4
# =====================================================

p4 = plot(
    diste4_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Orden"
)

plot!(
    dista4_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# PANEL FINAL
# =====================================================

p5=plot(
    p1, p2, p3, p4,
    layout=(2,2),
    size=(1000,700)
)

display(p5)

# =====================================================
# DISTANCIAS ACEPTADAS ORDENADAS
# =====================================================

# ==========================================
# Ordenar distancias aceptadas
# ==========================================

diste1_ord = sort(diste1[idx_euc1_aceptados])
dista1_ord = sort(dista1[idx_abs1_aceptados])

diste2_ord = sort(diste2[idx_euc2_aceptados])
dista2_ord = sort(dista2[idx_abs2_aceptados])

diste3_ord = sort(diste3[idx_euc3_aceptados])
dista3_ord = sort(dista3[idx_abs3_aceptados])

diste4_ord = sort(diste4[idx_euc4_aceptados])
dista4_ord = sort(dista4[idx_abs4_aceptados])


# =====================================================
# ABC 1
# =====================================================

p1 = plot(
    diste1_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Original"
)

plot!(
    dista1_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# ABC 2
# =====================================================

p2 = plot(
    diste2_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Momentos"
)

plot!(
    dista2_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# ABC 3
# =====================================================

p3 = plot(
    diste3_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Robusto"
)

plot!(
    dista3_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# ABC 4
# =====================================================

p4 = plot(
    diste4_ord,
    linewidth=2,
    label="Euclidiana",
    xlabel="Índice ordenado",
    ylabel="Distancia",
    title="ABC Orden"
)

plot!(
    dista4_ord,
    linewidth=2,
    label="Absoluta"
)


# =====================================================
# PANEL FINAL
# =====================================================

p6=plot(
    p1, p2, p3, p4,
    layout=(2,2),
    size=(1000,700)
)

display(p6)

# =====================================================
# ESTIMACIÓN NO PARAMÉTRICA
# DISTANCIAS EUCLIDIANAS
# =====================================================

θ_euc1 = densprob(θ_euc1_aceptados, 30)
θ_euc2 = densprob(θ_euc2_aceptados, 30)
θ_euc3 = densprob(θ_euc3_aceptados, 30)
θ_euc4 = densprob(θ_euc4_aceptados, 30)

# rango común

x_min_euc = min(θ_euc1.min, θ_euc2.min, θ_euc3.min, θ_euc4.min)

x_max_euc = max(θ_euc1.max, θ_euc2.max, θ_euc3.max, θ_euc4.max)

x_grid_euc = collect(range(x_min_euc, x_max_euc, length=400))

# densidades

y_euc1 = [θ_euc1.fdp(x) for x in x_grid_euc]
y_euc2 = [θ_euc2.fdp(x) for x in x_grid_euc]
y_euc3 = [θ_euc3.fdp(x) for x in x_grid_euc]
y_euc4 = [θ_euc4.fdp(x) for x in x_grid_euc]


# =====================================================
# GRÁFICO EUCLIDIANAS
# =====================================================

p7=plot(
    x_grid_euc,
    y_euc1,
    linewidth=3,
    label="ABC",
    xlabel="Distancia euclidiana",
    ylabel="Densidad",
    title="Estimación no paramétrica Theta: Euclidiana"
)

plot!(
    x_grid_euc,
    y_euc2,
    linewidth=3,
    label="Momentos"
)

plot!(
    x_grid_euc,
    y_euc3,
    linewidth=3,
    label="Robusto"
)

plot!(
    x_grid_euc,
    y_euc4,
    linewidth=3,
    label="Orden"
)

display(p7)

# =====================================================
# ESTIMACIÓN NO PARAMÉTRICA
# DISTANCIAS ABSOLUTAS
# =====================================================

θ_abs1 = densprob(θ_abs1_aceptados, 30)
θ_abs2 = densprob(θ_abs2_aceptados, 30)
θ_abs3 = densprob(θ_abs3_aceptados, 30)
θ_abs4 = densprob(θ_abs4_aceptados, 30)

# rango común

x_min_abs = min(θ_abs1.min, θ_abs2.min, θ_abs3.min, θ_abs4.min)

x_max_abs = max(θ_abs1.max, θ_abs2.max, θ_abs3.max, θ_abs4.max)

x_grid_abs = collect(range(x_min_abs, x_max_abs, length=400))

# densidades

y_abs1 = [θ_abs1.fdp(x) for x in x_grid_abs]
y_abs2 = [θ_abs2.fdp(x) for x in x_grid_abs]
y_abs3 = [θ_abs3.fdp(x) for x in x_grid_abs]
y_abs4 = [θ_abs4.fdp(x) for x in x_grid_abs]


# =====================================================
# GRÁFICO ABSOLUTAS
# =====================================================

p8=plot(
    x_grid_abs,
    y_abs1,
    linewidth=3,
    label="ABC",
    xlabel="Distancia absoluta",
    ylabel="Densidad",
    title="Estimación no paramétrica theta: Absoluta"
)

plot!(
    x_grid_abs,
    y_abs2,
    linewidth=3,
    label="Momentos"
)

plot!(
    x_grid_abs,
    y_abs3,
    linewidth=3,
    label="Robusto"
)

plot!(
    x_grid_abs,
    y_abs4,
    linewidth=3,
    label="Orden"
)

display(p8)

# =====================================================
# ESTIMACIÓN NO PARAMÉTRICA
# PARÁMETRO n
# DISTANCIA EUCLIDIANA
# =====================================================

m_neuc1 = masaprob(n_euc1_aceptados)
m_neuc2 = masaprob(n_euc2_aceptados)
m_neuc3 = masaprob(n_euc3_aceptados)
m_neuc4 = masaprob(n_euc4_aceptados)

# soporte común

n_vals_euc = sort(unique(vcat(
    m_neuc1.valores,
    m_neuc2.valores,
    m_neuc3.valores,
    m_neuc4.valores
)))

# probabilidades

p_neuc1 = [m_neuc1.fmp(n) for n in n_vals_euc]
p_neuc2 = [m_neuc2.fmp(n) for n in n_vals_euc]
p_neuc3 = [m_neuc3.fmp(n) for n in n_vals_euc]
p_neuc4 = [m_neuc4.fmp(n) for n in n_vals_euc]


# =====================================================
# GRÁFICO n — EUCLIDIANA
# =====================================================

p9=bar(
    n_vals_euc,
    p_neuc1,
    alpha=0.5,
    label="ABC",
    xlabel="n",
    ylabel="Probabilidad",
    title="Posterior de n: distancia euclidiana"
)

bar!(
    n_vals_euc,
    p_neuc2,
    alpha=0.5,
    label="Momentos"
)

bar!(
    n_vals_euc,
    p_neuc3,
    alpha=0.5,
    label="Robusto"
)

bar!(
    n_vals_euc,
    p_neuc4,
    alpha=0.5,
    label="Orden"
)

display(p9)

# =====================================================
# ESTIMACIÓN NO PARAMÉTRICA
# PARÁMETRO n
# DISTANCIA ABSOLUTA
# =====================================================

m_nabs1 = masaprob(n_abs1_aceptados)
m_nabs2 = masaprob(n_abs2_aceptados)
m_nabs3 = masaprob(n_abs3_aceptados)
m_nabs4 = masaprob(n_abs4_aceptados)

# soporte común

n_vals_abs = sort(unique(vcat(
    m_nabs1.valores,
    m_nabs2.valores,
    m_nabs3.valores,
    m_nabs4.valores
)))

# probabilidades

p_nabs1 = [m_nabs1.fmp(n) for n in n_vals_abs]
p_nabs2 = [m_nabs2.fmp(n) for n in n_vals_abs]
p_nabs3 = [m_nabs3.fmp(n) for n in n_vals_abs]
p_nabs4 = [m_nabs4.fmp(n) for n in n_vals_abs]


# =====================================================
# GRÁFICO n — ABSOLUTA
# =====================================================

p10=bar(
    n_vals_abs,
    p_nabs1,
    alpha=0.5,
    label="ABC",
    xlabel="n",
    ylabel="Probabilidad",
    title="Posterior de n: distancia absoluta"
)

bar!(
    n_vals_abs,
    p_nabs2,
    alpha=0.5,
    label="Momentos"
)

bar!(
    n_vals_abs,
    p_nabs3,
    alpha=0.5,
    label="Robusto"
)

bar!(
    n_vals_abs,
    p_nabs4,
    alpha=0.5,
    label="Orden"
)

display(p10)