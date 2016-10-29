### Expected arguments

### files
base_loc = "dump/data/".run_id."_"
plot_loc = "dump/plots/".run_id."_"
ext = ".dat"

cost_files = system("ls ".base_loc."cost_"."*".ext)
params_files = system("ls ".base_loc."params_"."*".ext)
weight_file = system("ls ".plot_loc."*weights"."*".ext)

params_names = system("ls ".base_loc."params_"."*".ext." | sed -e 's#".base_loc."params_"."\\(.*\\)\\".ext."#\\1#' -e 's#_# #g'")

#function used to map a value to the intervals
hist(x,width)=width*floor(x/width)+width/2.0

### Start multiplot
set terminal png size 1440,800 font 'Verdana,6'
set output "dump/visual_example_".iter.".png"

set multiplot
set autoscale x
xn=4
yn = 1+1+2
x = xn
y = yn
o = 0.1

xi = o
yi = o

xf = x-o
xs = xf-xi
yf = y-o
ys = yf-yi

# --- GRAPH cost
unset key
set title "cost vs. training samples" font ",8"
f = word(cost_files, 1)
set size (xn/2)*(xs/xn)/x,ys/(yn/2)/y
set origin xi/x,(yi+ys/(yn/2))/y
set xlabel "sample" font ",8"
set ylabel "value" font ",8"
plot f u (column(0)):1 w l ls 1 lc rgb"blue"

# --- GRAPH input image
unset key
set title "input weights" font ",8"
f = word(weight_file, 1)
set size (xn/2)*(xs/xn)/x,ys/(yn/2)/y
set origin (xi+(xn/2)*(xs/xn))/x,(yi+ys/(yn/2))/y
set border linewidth 0
unset key
unset colorbox
unset tics
unset xlabel
unset ylabel
set palette grey
plot f matrix w image

set xtics rotate by -45
# --- GRAPH params per batch
do for [i=1:words(params_files)] {
    unset key
    set tics
    p = word(params_names,i*2-1)." ".word(params_names,i*2)
    f = word(params_files, i)
    set title p." vs. training samples" font ",6"
    set size (xs/xn)/x,(ys/yn)/y
    set origin (xi+(i-1)*(xs/xn))/x,(yi+ys/yn)/y
    set xlabel "sample" font ",8"
    set ylabel "mean value" font ",8"
    plot f u (column(0)):1 with lines ls 1
}

# --- GRAPH histogram of params
do for [i=1:words(params_files)] {
    unset key
    set tics
    p = word(params_names,i*2-1)." ".word(params_names,i*2)
    f = word(params_files, i)
    
    set title "freq. of av. ".p font ",6"
    set tics
    set size (xs/xn)/x,(ys/yn)/y
    set origin (xi+(i-1)*(xs/xn))/x,yi/y
    stats f using 1 nooutput
    
    n=50 #number of intervals
    max=STATS_mean+3*STATS_stddev #max value
    min=STATS_mean-3*STATS_stddev #min value
    width=(max-min)/n #interval width
    
    # set boxwidth width*0.9
    # set style fill solid 0.5
    set tics
    set xlabel "mean value" font ",8"
    set ylabel "freq" font ",8"
    plot f u (hist($1,width)):(1.0/STATS_records) \
        smooth freq \
        w l \
        lc rgb"green" notitle
}

unset multiplot
### End multiplot

# pause 1
# reread