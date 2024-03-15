;首先我们需要一个读取函数，这里我直接复制过来讲解
function era5_readarr,file_name,sds_name ;这里只需要给函数传递文件名，和数据集名字
  file_id=ncdf_open(file_name) ;打开文件
  data_id=ncdf_varid(file_id,sds_name);读取数据集-我们这里是2m温度
  ncdf_varget,file_id,data_id,data;获取2m温度 存储为data
  ncdf_attget,file_id,data_id,'scale_factor',sc_data ;获取数据需要的预处理乘法因子，每个数据需要先乘以这个因子
  ncdf_attget,file_id,data_id,'add_offset',ao_data;相应的获取加法因子，每个数据需要乘以上面因子之后再加
  ncdf_close,file_id
  data=float(data)*sc_data+ao_data ;先乘后加
  return,data ;得到的最后数据返回
end

function era5_readarr11,file_name,sds_name ;这里只需要给函数传递文件名，和数据集名字
  file_id=ncdf_open(file_name) ;打开文件
  data_id=ncdf_varid(file_id,sds_name);读取数据集-我们这里是2m温度
  ncdf_varget,file_id,data_id,data;获取2m温度 存储为data
  ;ncdf_attget,file_id,data_id,'scale_factor',sc_data ;获取数据需要的预处理乘法因子，每个数据需要先乘以这个因子
  ;ncdf_attget,file_id,data_id,'add_offset',ao_data;相应的获取加法因子，每个数据需要乘以上面因子之后再加
  ncdf_close,file_id
  data=float(data);*sc_data+ao_data ;先乘后加
  return,data ;得到的最后数据返回
end
;接着我们对应我们的数据处理
pro sp_ncdf 
  ;输入文件名字直接复制过来也可以直接拖过来。
  path = 'F:\new_lunwen\Global GLDAS\'
  out_path = 'F:\new_lunwen\jg\Tveg\'
  file_dir = file_test(out_path,/directory)
  if file_dir eq 0 then begin
    file_mkdir,out_path
  endif
  ;接着我们读取数据-t2m
  file_list = file_search(path,'*.nc4',count = file_n)
  for file_i=0,file_n-1 do begin
    
    data_ciwc = era5_readarr11(file_list[file_i],'Tveg_tavg')
    data_ciwc = (data_ciwc gt -9999 and data_ciwc lt 9999)*data_ciwc
    
    lon = era5_readarr11(file_list[file_i],'lon')
    lon_min = min(lon)
    lat = era5_readarr11(file_list[file_i],'lat')
    lat_max = max(lat)
    data_ciwc = rotate(data_ciwc,7)
    
    res = 0.25
    geo_info={$
      MODELPIXELSCALETAG:[res,res,0.0],$  ;还是直接复制过来这是地理信息直接复制即可，这一行需要加入经纬度分辨率，都是0.25所以不用改
      MODELTIEPOINTTAG:[0.0,0.0,0.0,lon_min,lat_max,0.0],$ 
      GTMODELTYPEGEOKEY:2,$
      GTRASTERTYPEGEOKEY:1,$
      GEOGRAPHICTYPEGEOKEY:4326,$
      GEOGCITATIONGEOKEY:'GCS_WGS_1984',$
      GEOGANGULARUNITSGEOKEY:9102}
    ;写成tif
    write_tiff,out_path+file_basename(file_list[file_i],'.nc4')+'.tif',data_ciwc,/float,geotiff=geo_info ;输出路径out_path,文件名字2021——t2m.tif
    ;完成读取
    print,'down!!'
    
  endfor
end