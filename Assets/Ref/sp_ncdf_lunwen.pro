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
pro sp_ncdf_lunwen
  ;输入文件名字直接复制过来也可以直接拖过来。
  path = 'F:\new_lunwen\Global GLDAS\'
  
  out_path = 'F:\new_lunwen\Rainf_f_tavg\'
  out_path1 = 'F:\new_lunwen\Evap_tavg_tavg\'
  out_path2 = 'F:\new_lunwen\Qs_acc\'
  out_path3 = 'F:\new_lunwen\Qsb_acc\'
  if ~file_test(out_path1, /directory) then file_mkdir,out_path1
  if ~file_test(out_path2, /directory) then file_mkdir,out_path2
  if ~file_test(out_path3, /directory) then file_mkdir,out_path3
  file_dir = file_test(out_path,/directory)
  if file_dir eq 0 then begin
    file_mkdir,out_path
  endif 
  ;接着我们读取数据-t2m
  file_list = file_search(path,'*.nc4',count = file_n)
  for file_i=0,file_n-1 do begin

    year = fix(strmid(file_basename(file_list[file_i]),17,4))
    month = fix(strmid(file_basename(file_list[file_i]),21,2))
    data_ciwc = era5_readarr11(file_list[file_i],'Rainf_f_tavg')
    data_ciwc1 = era5_readarr11(file_list[file_i],'Evap_tavg')
    data_ciwc2 = era5_readarr11(file_list[file_i],'Qs_acc')
    data_ciwc3 = era5_readarr11(file_list[file_i],'Qsb_acc')



    data_ciwc = (data_ciwc gt -9999 and data_ciwc lt 9999)*data_ciwc
    data_ciwc1 = (data_ciwc1 gt -9999 and data_ciwc1 lt 9999)*data_ciwc1
    data_ciwc2 = (data_ciwc2 gt -9999 and data_ciwc2 lt 9999)*data_ciwc2
    data_ciwc3 = (data_ciwc3 gt -9999 and data_ciwc3 lt 9999)*data_ciwc3

    lon = era5_readarr11(file_list[file_i],'lon')
    lon_min = min(lon)
    lat = era5_readarr11(file_list[file_i],'lat')
    lat_max = max(lat)
    data_ciwc = rotate(data_ciwc,7)
    data_ciwc1 = rotate(data_ciwc1,7)
    data_ciwc2 = rotate(data_ciwc2,7)
    data_ciwc3 = rotate(data_ciwc3,7)
    if month eq 1 or month eq 3 or month eq 5 or month eq 7 or month eq 8 or month eq 10 or month eq 12 then begin
      data_ciwc = data_ciwc * 31 * 24 * 3600
      data_ciwc1 = data_ciwc1 * 31 * 24 * 3600
      data_ciwc2 = data_ciwc2 * 31 * 8
      data_ciwc3 = data_ciwc3 * 31 * 8
      
    endif
    if month eq 4 or month eq 6 or month eq 9 or month eq 11  then begin
      data_ciwc = data_ciwc * 30 * 24 * 3600
      data_ciwc1 = data_ciwc1 * 30 * 24 * 3600
      data_ciwc2 = data_ciwc2 * 30 * 8
      data_ciwc3 = data_ciwc3 * 30 * 8

    endif
    m = year mod 4 
    if month eq 2 and m eq 0 then begin
      data_ciwc = data_ciwc * 29 * 24 * 3600
      data_ciwc1 = data_ciwc1 * 29 * 24 * 3600
      data_ciwc2 = data_ciwc2 * 29 * 8
      data_ciwc3 = data_ciwc3 * 29 * 8
           
    endif
    if month eq 2 and m ne 0 then begin
      data_ciwc = data_ciwc * 28 * 24 * 3600
      data_ciwc1 = data_ciwc1 * 28 * 24 * 3600
      data_ciwc2 = data_ciwc2 * 28 * 8
      data_ciwc3 = data_ciwc3 * 28 * 8

    endif
      
    res = 0.25
    geo_info={$
      MODELPIXELSCALETAG:[res,res,0.0],$  ;还是直接复制过来这是地理信息直接复制即可，这一行需要加入经纬度分辨率，都是0.25所以不用改
      MODELTIEPOINTTAG:[0.0,0.0,0.0,lon_min,lat_max,0.0],$ ;这里需要提供最小经度和最大纬度，在第4、5个位置
      GTMODELTYPEGEOKEY:2,$
      GTRASTERTYPEGEOKEY:1,$
      GEOGRAPHICTYPEGEOKEY:4326,$
      GEOGCITATIONGEOKEY:'GCS_WGS_1984',$
      GEOGANGULARUNITSGEOKEY:9102}
    ;写成tif
    write_tiff,out_path+file_basename(file_list[file_i],'.nc4')+'.tif',data_ciwc,/float,geotiff=geo_info ;输出路径out_path,文件名字2021——t2m.tif
    write_tiff,out_path1+file_basename(file_list[file_i],'.nc4')+'.tif',data_ciwc1,/float,geotiff=geo_info
    write_tiff,out_path2+file_basename(file_list[file_i],'.nc4')+'.tif',data_ciwc2,/float,geotiff=geo_info
    write_tiff,out_path3+file_basename(file_list[file_i],'.nc4')+'.tif',data_ciwc3,/float,geotiff=geo_info
    ;完成读取
    print,'down!!'

  endfor
end