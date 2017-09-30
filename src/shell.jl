import Images
import FileIO
import ImageView

function doitall()
    include("microtubule_asters.jl")
    img, lab = genvor.load_state()
    @time pq = genvor.generalized_voronoi(lab)
    @code_warntype genvor.generalized_voronoi(lab)
    @time vor, pq2 = genvor.grow_regions(lab, img, pq, 10000)
    @code_warntype genvor.grow_regions(lab, img, pq, 10000)
end

function qsave(img)
    Images.save("qsave.tif", convert(Array{UInt8}, img));
    run(`open qsave.tif`);
end
