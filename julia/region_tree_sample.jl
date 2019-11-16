using RegionTrees
using StaticArrays

function gen_sample_quadtree()
    root = Cell(SVector(0, 0), SVector(1, 1))
    cells = [root]
    for i in 1:6
        cell_current = begin
            len = length(cells)
            if len == 1
                root
            elseif len == 5
                cells[end]
            else
                cells[end-3]
            end
        end
        
        (length(cells)==1 ? root : cells[end-1])
        split!(cell_current)
        map((c)->push!(cells, c), children(cell_current))
    end
    return root
end

root = gen_sample_quadtree()

using Plots
plt = plot(xlim=(0, 1), ylim=(0, 1), legend=nothing)

for leaf in allleaves(root)
    v = hcat(collect(vertices(leaf.boundary))...)
    plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], color=:black)
end

plt
