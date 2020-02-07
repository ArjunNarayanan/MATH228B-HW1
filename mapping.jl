using BenchmarkTools
using LinearAlgebra, PyPlot, SparseArrays, Statistics, Printf

N1(xi,eta) = +(xi - 1.0)*(eta - 1.0)/4.0
dN1(xi,eta) = 0.25*[(eta-1.0)  (xi-1.0)]
d2N1(xi,eta) = [0.0 0.25
                0.25 0.0]

N2(xi,eta) = -(xi + 1.0)*(eta - 1.0)/4.0
dN2(xi,eta) = -0.25*[(eta-1.0)  (xi+1.0)]
d2N2(xi,eta) = [0.0  -0.25
               -0.25 0.0]

N3(xi,eta) = +(xi + 1.0)*(eta + 1.0)/4.0
dN3(xi,eta) = 0.25*[(eta+1.0)  (xi+1.0)]
d2N3(xi,eta) = [0.0 0.25
                0.25 0.0]

N4(xi,eta) = -(xi - 1.0)*(eta + 1.0)/4.0
dN4(xi,eta) = -0.25*[(eta+1.0)  (xi-1.0)]
d2N4(xi,eta) = [0.0 -0.25
               -0.25 0.0]

function corners(B,L,H)
    D = (L - B)/2.0
    theta = asin(H/D)
    A = D*cos(theta)
    points = [0.0   B/2     (B/2+A)     0.0
              0.0   0.0     H           H]
    return points
end

function basis(xi,eta)
    return [N1(xi,eta), N2(xi,eta), N3(xi,eta), N4(xi,eta)]
end

function gradient(xi,eta)
    return vcat(dN1(xi,eta),dN2(xi,eta),dN3(xi,eta),dN4(xi,eta))
end

function hessian(xi,eta)
    return [d2N1(xi,eta), d2N2(xi,eta), d2N3(xi,eta), d2N4(xi,eta)]
end

function mul_point_hessian(p,NH)
    h = zeros(2,2,2)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                h[i,j,k] = p[i]*NH[j,k]
            end
        end
    end
    return h
end

function jacobian(points::Matrix, xi,eta)
    return points*gradient(xi,eta)
end

function hessian(points::Matrix, xi, eta)
    h = zeros(2,2,2)
    NH = hessian(xi,eta)
    for i in 1:size(points)[2]
        h = h + mul_point_hessian(view(points,:,i),NH[i])
    end
    return h
end

function determinant(jac::Matrix)
    return jac[1,1]*jac[2,2] - jac[1,2]*jac[2,1]
end

function pullback(jac::Matrix, spatial_normal::Vector)
    return [jac[2,2]*spatial_normal[1] - jac[1,2]*spatial_normal[2],
           -jac[2,1]*spatial_normal[1] + jac[1,1]*spatial_normal[2]]
end

function diffeq_coefficients(jac::Matrix,hess::Array{T,3},J) where {T}
    a = jac[1,2]^2 + jac[2,2]^2
    b = jac[1,1]*jac[1,2] + jac[2,1]*jac[2,2]
    c = jac[1,1]^2 + jac[2,1]^2
    alpha = a*hess[1,1,1] - 2*b*hess[1,1,2] + c*hess[1,2,2]
    beta  = a*hess[2,1,1] - 2*b*hess[2,1,2] + c*hess[2,2,2]
    d = (jac[1,2]*beta - jac[2,2]*alpha)/J
    e = (jac[2,1]*alpha - jac[1,1]*beta)/J
    return a,b,c,d,e
end

function stencilCoefficients(jac::Matrix, hess::Array{T,3},J,h) where {T}
    a,b,c,d,e = diffeq_coefficients(jac,hess,J)
    J2 = J^2
    C1 = -a/J2
    C2 = 2b/J2
    C3 = -c/J2
    C4 = -d/J2
    C5 = -e/J2

    h2 = h^2
    V1 = C2/(4h2)
    V2 = (C1/h2 + C4/(2h))
    V3 = -C2/(4h2)
    V4 = (C3/h2 + C5/(2h))
    V5 = -2/h2*(C1 + C3)
    V6 = (C3/h2 - C5/(2h))
    V7 = -C2/(4h2)
    V8 = (C1/h2 - C4/(2h))
    V9 = C2/(4h2)

    return V1, V2, V3, V4, V5, V6, V7, V8, V9
end

function interpolate(vals::Matrix, xi, eta)
    return vals*basis(xi,eta)
end

function map_to_spatial(xrange::AbstractVector, spatial_corners::Matrix)
    num_points_1d = length(xrange)
    num_points = num_points_1d^2
    points = zeros(2,num_points)
    count = 1
    for x in xrange
        for y in xrange
            p = interpolate(spatial_corners, x, y)
            points[:,count] = p
            count += 1
        end
    end
    return points
end

function indexToDOF(i::Int,j::Int,N::Int)
    return (j-1)*N + i
end

function onBoundary(i::Int,j::Int,N::Int)
    if j == 1 || j == N || i == 1 || i == N
        return true
    else
        return false
    end
end

function onLeftBoundary(i::Int,j::Int,N::Int)
    return (j == 1 && i > 1 && i < N ? true : false)
end

function onLeftCorner(i,j,N)
    return (j == 1 && i == N ? true : false)
end

function onCorners(i,j,N)
    if (i == 1 || i == N) && (j == 1 || j == N)
        return true
    else
        return false
    end
end

function onRightBoundary(i,j,N)
    return (j == N ? true : false)
end

function onTopBoundary(i,j,N)
    return (i == N ? true : false)
end

function onBottomBoundary(i,j,N)
    return (i == 1 ? true : false)
end

function indexToDOF(idx::Vector{Tuple{Int,Int}},N)
    return [indexToDOF(v[1],v[2],N) for v in idx]
end

function stencilIdx(i,j)
    return [(p,q) for q in j+1:-1:j-1 for p in i+1:-1:i-1]
end

function stencilLeftBoundary(i,j)
    return [(i,j+2), (i,j+1), (i+1,j), (i,j), (i-1,j)]
end

function stencilLeftCorner(i,j)
    [(i,j+2), (i,j+1), (i,j), (i-1,j), (i-2,j)]
end

function stencilCoefficientsLeftCorner(n::Vector)
    return [-1.0n[1], 4n[1], (-3n[1]+3n[2]), -4n[2], n[2]]
end

function stencilCoefficientsLeftBoundary(n::Vector)
    return [-1.0n[1], 4n[1], n[2], -3n[1], -n[2]]
end

function stencilTopBoundary(i,j)
    return [(i,j+1), (i,j), (i,j-1), (i-1,j), (i-2,j)]
end

function stencilCoefficientsTopBoundary(n::Vector)
    return [1.0n[1], 3n[2], -n[1], -4n[2], n[2]]
end

function assemblePoisson(N::Int, spatial_corners::Matrix)
    xrange = range(-1.0, stop = 1.0, length = N)
    dx = xrange[2] - xrange[1]

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:N
        for i in 1:N
            r = indexToDOF(i,j,N)
            xi = xrange[j]
            eta = xrange[i]
            jac = jacobian(spatial_corners,xi,eta)
            J = determinant(jac)
            if !onBoundary(i,j,N)
                col = indexToDOF(stencilIdx(i,j),N)
                hess = hessian(spatial_corners, xi, eta)
                coeffs = stencilCoefficients(jac,hess,J,dx)
                row = repeat([r],9)
                append!(rows, row)
                append!(cols, col)
                append!(vals, coeffs)
            elseif onBottomBoundary(i,j,N) || onRightBoundary(i,j,N)
                append!(rows, r)
                append!(cols, r)
                append!(vals, 1.0)
            elseif onLeftBoundary(i,j,N)
                col = indexToDOF(stencilLeftBoundary(i,j),N)
                spatial_normal = [-1.0,0.0]
                ref_normal = pullback(jac, spatial_normal)
                coeffs = stencilCoefficientsLeftBoundary(ref_normal)
                row = repeat([r],5)
                append!(rows,row)
                append!(cols,col)
                append!(vals,coeffs)
            elseif onLeftCorner(i,j,N)
                col = indexToDOF(stencilLeftCorner(i,j),N)
                spatial_normal = [-1.0,0.0]
                ref_normal = pullback(jac, spatial_normal)
                coeffs = stencilCoefficientsLeftCorner(ref_normal)
                row = repeat([r],5)
                append!(rows,row)
                append!(cols,col)
                append!(vals,coeffs)
            elseif onTopBoundary(i,j,N)
                col = indexToDOF(stencilTopBoundary(i,j),N)
                spatial_normal = [0.0,1.0]
                ref_normal = pullback(jac, spatial_normal)
                coeffs = stencilCoefficientsTopBoundary(ref_normal)
                row = repeat([r],5)
                append!(rows,row)
                append!(cols,col)
                append!(vals,coeffs)
            end
        end
    end
    A = sparse(rows,cols,vals)
    return A, xrange
end

function assembleRHS(N::Int)
    ndofs = N^2
    rhs = zeros(ndofs)
    for j in 1:N
        for i in 1:N
            r = indexToDOF(i,j,N)
            if !onBoundary(i,j,N)
                rhs[r] = 1.0
            end
        end
    end
    return rhs
end

function solveProblem(L,B,H,N)
    dx = 2.0/(N - 1)
    spatial_corners = corners(B,L,H)
    A, xrange = assemblePoisson(N, spatial_corners)
    rhs = assembleRHS(N)
    sol = A\rhs
    Q = simpson_integrate(sol,N,dx)

    return sol, Q
end

function simpson_integrate(sol::Vector, N::Int, dx)
    integral = 0.0
    for i in 1:N
        for j in 1:N
            r = indexToDOF(i,j,N)
            if onCorners(i,j,N)
                integral += 0.25*sol[r]
            elseif onBoundary(i,j,N)
                integral += 0.5*sol[r]
            else
                integral += sol[r]
            end
        end
    end
    return dx^2*integral
end

const L = 3.0
const B = 0.5
const H = 1.0

spatial_corners = corners(B,L,H)
N = 80
dx = 2.0/(N-1)

Nrange = [10,20,40,80]
hrange = [2.0/(N-1) for N in Nrange]
Qrange = [solveProblem(L,B,H,N)[2] for N in Nrange]
Qrange = [abs(q - Qrange[end]) for q in Qrange]
fig, ax = PyPlot.subplots()
ax.loglog(hrange[1:end-1],Qrange[1:end-1],"-o")
ax.grid()
s = mean(diff(log.(Qrange[1:end-1])) ./ diff(log.(hrange[1:end-1])))
msg = @sprintf "Mean slope = %1.2f" s
ax.annotate(msg, (0.5,0.3), xycoords = "axes fraction")
fig

# xrange = range(-1, stop = 1, length = N)
# sol,Q = solveProblem(L,B,H,N)
# Q1 = sum(sol)*dx^2
# println("Difference = ", abs(Q - Q1))
# spatial_points = map_to_spatial(xrange, spatial_corners)
# X = reshape(spatial_points[1,:], N, N)
# Y = reshape(spatial_points[2,:], N, N)
# U = reshape(sol, N, N)
#
# fig, ax = PyPlot.subplots()
# cp = ax.contour(X, Y, U, levels = collect(0.0:0.02:0.2))
# ax.clabel(cp, inline=true, fontsize=10)
# ax.set_aspect(1.0)
# fig
