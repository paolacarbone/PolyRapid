from dl_field_wrapper.topology_builder import TopologyBuilder, batch_build



# Provide path to DL_FIELD root directory to initialise 
builder = TopologyBuilder(dl_field_path="/home/lois181/dl_f_4.12/") # path to dl_field exe

custom_control = "/home/lois181/dl_f_4.11/polymer.control" # control file path  
root_dir = "/home/lois181/code/try-chains-make/identifying-errors/Output" # directory where the xyz files are 



batch_build(builder, root_dir, custom_control) 









