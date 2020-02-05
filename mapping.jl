using LinearAlgebra, Plots

N1(xi,eta) = +(xi - 1.0)*(eta - 1.0)/4.0
dN1(xi,eta) = 0.25*[(eta-1.0)  (xi-1.0)]

N2(xi,eta) = -(xi + 1.0)*(eta - 1.0)/4.0
dN2(xi,eta) = -0.25*[(eta-1.0)  (xi+1.0)]

N3(xi,eta) = +(xi + 1.0)*(eta + 1.0)/4.0
dN3(xi,eta) = 0.25*[(eta+1.0)  (xi+1.0)]

N4(xi,eta) = -(xi - 1.0)*(eta + 1.0)/4.0
dN4(xi,eta) = -0.25*[(eta+1.0)  (xi-1.0)]

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

function jacobian(points::Matrix, xi,eta)
    return points*gradient(xi,eta)
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
    return (j == 1 ? true : false)
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
    return [(p,q) for p in i-1:i+1 for q in j-1:j+1]
end

function stencilCoefficients(M::Matrix,dx)
    mid = (M[1,2] + M[2,1])/(2.0*dx)
    ymax = M[2,2]/dx^2
    xmax = M[1,1]/dx^2
    return [mid, ymax, -mid, xmax, -2.0*(xmax+ymax), xmax, -mid, ymax, mid]
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
            if !onBoundary(i,j,N)
                col = indexToDOF(stencilIdx(i,j),N)
                inverse_jacobian = inv(jacobian(spatial_corners, xrange[j], xrange[i]))
                M = inverse_jacobian'*inverse_jacobian
                coeffs = stencilCoefficients(M,dx)
                row = repeat([r],9)
                append!(rows, row)
                append!(cols, col)
                append!(vals, coeffs)
            elseif onBottomBoundary(i,j,N) || onRightBoundary(i,j,N)
                append!(rows, r)
                append!(cols, r)
                append!(vals, 1.0)
            elseif onLeftBoundary(i,j,N)

            end
        end
    end
    return rows, cols, vals
end

const L = 3.0
const B = 0.5
const H = 1.0

const N = 3
const dx = 2.0/(N - 1)
spatial_corners = corners(B,L,H)

rows, cols, vals = assemblePoisson(N, spatial_corners)

# xrange = -1:dx:1
# points = map_to_spatial(xrange, spatial_corners)
# scatter(points[1,:], points[2,:], aspect_ratio = 1.0)
