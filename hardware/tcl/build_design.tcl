# Vivado 2025.2 — KV260 block design around the crisp_top HLS IP, bitstream, reports.  Run by hardware/build.py from vivado/:
#   vivado -mode batch -source build_bd.tcl -log build_<variant>.log -journal build_<variant>.jou
# Environment (set by hardware/build.py): CRISP_VARIANT (sampled|mf), CRISP_IP_DIR (packaged HLS IP: <work>/hls/impl/ip),
#   CRISP_JOBS, CRISP_PL_MHZ (default 100; the PS picks the nearest divider of 1499.985 MHz, e.g. 99.999 MHz).
# Outputs: crisp_<variant>_vivado/..../design_1_wrapper.bit, reports_<variant>/*.rpt + timing_status.txt, crisp_<variant>.bif,
#   design_<variant>.xsa, ../firmware_<variant>/pl.dtsi + pl_clk_hz.txt (overlay with the PL clock actually constrained)
set here [file dirname [file normalize [info script]]]
cd $here
set part    xck26-sfvc784-2LV-c
set variant [expr {[info exists ::env(CRISP_VARIANT)] ? $::env(CRISP_VARIANT) : "sampled"}]
set jobs    [expr {[info exists ::env(CRISP_JOBS)] ? $::env(CRISP_JOBS) : 8}]
set pl_mhz  [expr {[info exists ::env(CRISP_PL_MHZ)] ? $::env(CRISP_PL_MHZ) : 100}]
if {![info exists ::env(CRISP_IP_DIR)]} { error "Run hardware/build.py so the validated HLS IP path is selected." }
set ip_dir [file normalize $::env(CRISP_IP_DIR)]
if {![file exists $ip_dir/component.xml]} { error "Packaged HLS IP missing: $ip_dir" }
puts "\[BD\] variant $variant | IP repository: $ip_dir | PL clock request: $pl_mhz MHz"

set proj crisp_${variant}_vivado
create_project $proj ./$proj -part $part -force
set bp [lindex [get_board_parts -quiet -latest_file_version "*kv260_som*"] 0]
if {$bp ne ""} {
    set_property board_part $bp [current_project]
    catch { set_property board_connections [list som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:1.4] [current_project] }
    puts "\[BD\] board part: $bp"
} else {
    error "KV260 board part missing. Install the SOM/carrier board files before building."
}
set_property ip_repo_paths $ip_dir [current_project]
update_ip_catalog

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {$bp ne ""} { apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps }
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} CONFIG.PSU__USE__S_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_mhz \
    CONFIG.PSU__USE__IRQ0 {0} ] $ps
set pl_act [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
puts "\[BD\] PL clock 0: requested $pl_mhz MHz, actual (constraint) $pl_act MHz"

set ip [create_bd_cell -type ip -vlnv xilinx.com:hls:crisp_top:1.0 crisp_top_0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/crisp_top_0/s_axi_ctrl} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
    [get_bd_intf_pins crisp_top_0/s_axi_ctrl]
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] -filter {NAME =~ "*crisp_top*"}]
if {$seg ne ""} { set_property offset 0xA0000000 $seg; set_property range 64K $seg }
validate_bd_design
save_bd_design
set wrapper [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
open_run impl_1
set rep reports_${variant}
file mkdir $rep
report_utilization -hierarchical -hierarchical_depth 3 -file $rep/utilization_hier.rpt
report_utilization -file $rep/utilization.rpt
report_timing_summary -file $rep/timing.rpt
report_power -file $rep/power_vectorless.rpt
set bit [file normalize $proj/$proj.runs/impl_1/design_1_wrapper.bit]
write_hw_platform -fixed -include_bit -force design_${variant}.xsa

set run [get_runs impl_1]
set wns [get_property STATS.WNS $run]; set tns [get_property STATS.TNS $run]; set whs [get_property STATS.WHS $run]
if {![string is double -strict $wns] || ![string is double -strict $whs]} {
    set wns [get_property SLACK [lindex [get_timing_paths -setup -max_paths 1] 0]]
    set whs [get_property SLACK [lindex [get_timing_paths -hold  -max_paths 1] 0]]
    if {![string is double -strict $tns]} { set tns "n/a" }
}
set met [expr {$wns >= 0 && $whs >= 0}]
set verdict [expr {$met ? "MET" : "NOT MET"}]
set f [open $rep/timing_status.txt w]
puts $f "PL clock: $pl_act MHz (requested $pl_mhz)\nWNS = $wns ns\nTNS = $tns ns\nWHS = $whs ns\nTIMING $verdict"
close $f

set f [open crisp_${variant}.bif w]
puts $f "all:\n{\n  \[destination_device = pl\] $bit\n}"
close $f
set pl_hz [expr {int(floor($pl_act * 1e6)) - 1000}]
set fw [file normalize ../firmware_${variant}]
file mkdir $fw
set f [open $fw/pl_clk_hz.txt w]; fconfigure $f -translation lf; puts $f $pl_hz; close $f
set f [open $fw/pl.dtsi w]; fconfigure $f -translation lf
puts $f "/dts-v1/;
/plugin/;
/* Kria firmware overlay for the crisp_top ($variant) measurement design (written by vivado/build_bd.tcl): loads
   crisp_${variant}.bit.bin through the FPGA manager and sets PL clock 0 (zynqmp_clk 71 = pl0_ref) to $pl_hz Hz = the
   $pl_act MHz the bitstream was constrained at. The IP is reached via /dev/mem at 0xA000_0000 (AXI-Lite); no driver node. */
&fpga_full {
    firmware-name = \"crisp_${variant}.bit.bin\";
    resets = <&zynqmp_reset 116>, <&zynqmp_reset 117>, <&zynqmp_reset 118>, <&zynqmp_reset 119>;
};
&amba {
    afi0: afi0 {
        compatible = \"xlnx,afi-fpga\";
        config-afi = <0 0>, <1 0>, <2 0>, <3 0>, <4 0>, <5 0>, <6 0>, <7 0>, <8 0>, <9 0>, <10 0>, <11 0>, <12 0>, <13 0>, <14 0xa00>, <15 0x000>;
    };
    clocking0: clocking0 {
        #clock-cells = <0>;
        assigned-clock-rates = <$pl_hz>;
        assigned-clocks = <&zynqmp_clk 71>;
        clock-output-names = \"fabric_clk\";
        clocks = <&zynqmp_clk 71>;
        compatible = \"xlnx,fclk\";
    };
};"
close $f
puts "\n\[BD\] ================================================================"
puts "\[BD\] $variant: PL clock $pl_act MHz   WNS $wns ns   TNS $tns ns   WHS $whs ns   ->  TIMING $verdict"
if {!$met} { error "Timing failed; firmware packaging is blocked. Check $rep/timing.rpt (retry with CRISP_PL_MHZ=89 via hardware/build.py --pl-mhz 89)." }
puts "\[BD\] bitstream: $bit"
puts "\[BD\] ================================================================"
exit
