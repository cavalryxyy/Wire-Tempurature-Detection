from func import Compute

def workflow(df):
    res = Compute.extract_wire(df)  
    wire, region, wire_num = Compute.global_filter_small(res, threshold = 640)
    yloc = []
    r_max, xm_pos, t_min = [], [] ,[]
    
    for sub_region in region:
        _yloc = Compute.get_coor_pairs(sub_region)
        sub_wire = Compute.find_pos(wire, _yloc)
        _wire = Compute.local_filter_small(sub_wire)
        _r_max, _xm_pos, _t_min = Compute.loc_mp(_wire)
        
        yloc.append(_yloc)
        r_max.append(_r_max)
        xm_pos.append(_xm_pos)
        t_min.append(_t_min)
    
    return r_max, xm_pos, t_min, yloc