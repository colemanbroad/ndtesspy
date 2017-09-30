# microtubule_asters.jl

module genvor

# import DataStructures as ds # julia0.6.0
# import Collections as ds

ds = Collections
import Images

data = """
check out this shit
http://juliaimages.github.io/latest/
"""

# neighbor_grid = [-1 0; 1 0; 0 -1; 0 1;]
# neighbor_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]


function load_state()
    img = Images.load("microtubule foams.jpg")
    img = Images.raw(img)
    img = convert(Array{Float64}, img)
    lab = Images.load("lab.png")
    lab = Images.raw(lab)
    lab = convert(Array{Int64}, lab)
    img, lab
end 

function generalized_voronoi(vorimg)
    xmax, ymax = size(vorimg)
    pq = ds.PriorityQueue(VorPt, Float64)
    # pq = ds.PriorityQueue{VorPt, Float64}() # julia0.6.0
    for i in 1:xmax
        for j in 1:ymax
            l = vorimg[i,j]
            if l!=0
                d = 0.0
                vp = VorPt(d,i,j,l,i,j)
                ds.enqueue!(pq, vp, d)
            end
        end
    end
    pq
end

immutable VorPt
    d::Float64
    x::Int64
    y::Int64
    l::Int64
    px::Int64
    py::Int64
end

function grow_regions(lab, distimg, pq_init, nnn)
    neighbor_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    pq = deepcopy(pq_init)
    vorimg = deepcopy(lab)
    xmax, ymax = size(vorimg)
    s = []
    for _ in 1:nnn
        vp = ds.dequeue!(pq)
        # println(vp)
        d, x, y, l = vp.d, vp.x, vp.y, vp.l
        if vorimg[vp.x,vp.y]==0
            vorimg[vp.x,vp.y]=l
        end
        for (dx,dy) in neighbor_grid
            nx,ny = vp.x+dx, vp.y+dy
            # println(nx, ny)
            if 1 <= nx <= xmax
                if 1 <= ny <= ymax
                    nl = vorimg[nx,ny]
                    # println(nl)
                    if nl == 0
                        f1 = distimg[x,y]
                        f2 = distimg[x+dx, y+dy]
                        # nd = metric_tensor(x,y,dx,dy,f1,f2) + d
                        nd = 1.0 + d
                        vp = VorPt(nd, nx, ny, l, x, y)
                        try
                            ds.enqueue!(pq, vp, nd)
                        catch
                            push!(s, vp)
                        end
                    end
                end
            end
        end
    end
    vorimg, pq
end

function metric_tensor(x,y,dx,dy,f1,f2)
    # 1.0 #0.5 * (f1 + f2)*(dx^2 + dy^2)
end

end