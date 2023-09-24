// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo

//@version=5
indicator("Smart Money Concepts [LuxAlgo]", "Smart Money Concepts [LuxAlgo]"
  , overlay = true
  , max_labels_count = 500
  , max_lines_count = 500
  , max_boxes_count = 500
  , max_bars_back = 500)
//-----------------------------------------------------------------------------{
//Constants
//-----------------------------------------------------------------------------{
color TRANSP_CSS = #ffffff00

//Tooltips
string MODE_TOOLTIP          = 'Allows to display historical Structure or only the recent ones'
string STYLE_TOOLTIP         = 'Indicator color theme'
string COLOR_CANDLES_TOOLTIP = 'Display additional candles with a color reflecting the current trend detected by structure'
string SHOW_INTERNAL         = 'Display internal market structure'
string CONFLUENCE_FILTER     = 'Filter non significant internal structure breakouts'
string SHOW_SWING            = 'Display swing market Structure'
string SHOW_SWING_POINTS     = 'Display swing point as labels on the chart'
string SHOW_SWHL_POINTS      = 'Highlight most recent strong and weak high/low points on the chart'
string INTERNAL_OB           = 'Display internal order blocks on the chart\n\nNumber of internal order blocks to display on the chart'
string SWING_OB              = 'Display swing order blocks on the chart\n\nNumber of internal swing blocks to display on the chart'
string FILTER_OB             = 'Method used to filter out volatile order blocks \n\nIt is recommended to use the cumulative mean range method when a low amount of data is available'
string SHOW_EQHL             = 'Display equal highs and equal lows on the chart'
string EQHL_BARS             = 'Number of bars used to confirm equal highs and equal lows'
string EQHL_THRESHOLD        = 'Sensitivity threshold in a range (0, 1) used for the detection of equal highs & lows\n\nLower values will return fewer but more pertinent results'
string SHOW_FVG              = 'Display fair values gaps on the chart'
string AUTO_FVG              = 'Filter out non significant fair value gaps'
string FVG_TF                = 'Fair value gaps timeframe'
string EXTEND_FVG            = 'Determine how many bars to extend the Fair Value Gap boxes on chart'
string PED_ZONES             = 'Display premium, discount, and equilibrium zones on chart'

//-----------------------------------------------------------------------------{
//Settings
//-----------------------------------------------------------------------------{
//General
//----------------------------------------{
mode = input.string('Historical'
  , options = ['Historical', 'Present']
  , group = 'Smart Money Concepts'
  , tooltip = MODE_TOOLTIP)

style = input.string('Colored'
  , options = ['Colored', 'Monochrome']
  , group = 'Smart Money Concepts'
  , tooltip = STYLE_TOOLTIP)

show_trend = input(false, 'Color Candles'
  , group = 'Smart Money Concepts'
  , tooltip = COLOR_CANDLES_TOOLTIP)

//----------------------------------------}
//Internal Structure
//----------------------------------------{
show_internals = input(true, 'Show Internal Structure'
  , group = 'Real Time Internal Structure'
  , tooltip = SHOW_INTERNAL)

show_ibull = input.string('All', 'Bullish Structure'
  , options = ['All', 'BOS', 'CHoCH']
  , inline = 'ibull'
  , group = 'Real Time Internal Structure')

swing_ibull_css = input(#089981, ''
  , inline = 'ibull'
  , group = 'Real Time Internal Structure')

//Bear Structure
show_ibear = input.string('All', 'Bearish Structure'
  , options = ['All', 'BOS', 'CHoCH']
  , inline = 'ibear'
  , group = 'Real Time Internal Structure')

swing_ibear_css = input(#f23645, ''
  , inline = 'ibear'
  , group = 'Real Time Internal Structure')

ifilter_confluence = input(false, 'Confluence Filter'
  , group = 'Real Time Internal Structure'
  , tooltip = CONFLUENCE_FILTER)

internal_structure_size = input.string('Tiny', 'Internal Label Size'
  , options = ['Tiny', 'Small', 'Normal']
  , group = 'Real Time Internal Structure')

//----------------------------------------}
//Swing Structure
//----------------------------------------{
show_Structure = input(true, 'Show Swing Structure'
  , group = 'Real Time Swing Structure'
  , tooltip = SHOW_SWING)

//Bull Structure
show_bull = input.string('All', 'Bullish Structure'
  , options = ['All', 'BOS', 'CHoCH']
  , inline = 'bull'
  , group = 'Real Time Swing Structure')

swing_bull_css = input(#089981, ''
  , inline = 'bull'
  , group = 'Real Time Swing Structure')

//Bear Structure
show_bear = input.string('All', 'Bearish Structure'
  , options = ['All', 'BOS', 'CHoCH']
  , inline = 'bear'
  , group = 'Real Time Swing Structure')

swing_bear_css = input(#f23645, ''
  , inline = 'bear'
  , group = 'Real Time Swing Structure')

swing_structure_size = input.string('Small', 'Swing Label Size'
  , options = ['Tiny', 'Small', 'Normal']
  , group = 'Real Time Swing Structure')

//Swings
show_swings = input(false, 'Show Swings Points'
  , inline = 'swings'
  , group = 'Real Time Swing Structure'
  , tooltip = SHOW_SWING_POINTS)

length = input.int(50, ''
  , minval = 10
  , inline = 'swings'
  , group = 'Real Time Swing Structure')

show_hl_swings = input(true, 'Show Strong/Weak High/Low'
  , group = 'Real Time Swing Structure'
  , tooltip = SHOW_SWHL_POINTS)

//----------------------------------------}
//Order Blocks
//----------------------------------------{
show_iob = input(true, 'Internal Order Blocks'
  , inline = 'iob'
  , group = 'Order Blocks'
  , tooltip = INTERNAL_OB)

iob_showlast = input.int(5, ''
  , minval = 1
  , inline = 'iob'
  , group = 'Order Blocks')

show_ob = input(false, 'Swing Order Blocks'
  , inline = 'ob'
  , group = 'Order Blocks'
  , tooltip = SWING_OB)

ob_showlast = input.int(5, ''
  , minval = 1
  , inline = 'ob'
  , group = 'Order Blocks')

ob_filter = input.string('Atr', 'Order Block Filter'
  , options = ['Atr', 'Cumulative Mean Range']
  , group = 'Order Blocks'
  , tooltip = FILTER_OB)

ibull_ob_css = input.color(color.new(#3179f5, 80), 'Internal Bullish OB'
  , group = 'Order Blocks')

ibear_ob_css = input.color(color.new(#f77c80, 80), 'Internal Bearish OB'
  , group = 'Order Blocks')

bull_ob_css = input.color(color.new(#1848cc, 80), 'Bullish OB'
  , group = 'Order Blocks')

bear_ob_css = input.color(color.new(#b22833, 80), 'Bearish OB'
  , group = 'Order Blocks')

//----------------------------------------}
//EQH/EQL
//----------------------------------------{
show_eq = input(true, 'Equal High/Low'
  , group = 'EQH/EQL'
  , tooltip = SHOW_EQHL)

eq_len = input.int(3, 'Bars Confirmation'
  , minval = 1
  , group = 'EQH/EQL'
  , tooltip = EQHL_BARS)

eq_threshold = input.float(0.1, 'Threshold'
  , minval = 0
  , maxval = 0.5
  , step = 0.1
  , group = 'EQH/EQL'
  , tooltip = EQHL_THRESHOLD)

eq_size = input.string('Tiny', 'Label Size'
  , options = ['Tiny', 'Small', 'Normal']
  , group = 'EQH/EQL')

//----------------------------------------}
//Fair Value Gaps
//----------------------------------------{
show_fvg = input(false, 'Fair Value Gaps'
  , group = 'Fair Value Gaps'
  , tooltip = SHOW_FVG)
  
fvg_auto = input(true, "Auto Threshold"
  , group = 'Fair Value Gaps'
  , tooltip = AUTO_FVG)

fvg_tf = input.timeframe('', "Timeframe"
  , group = 'Fair Value Gaps'
  , tooltip = FVG_TF)

bull_fvg_css = input.color(color.new(#00ff68, 70), 'Bullish FVG'
  , group = 'Fair Value Gaps')

bear_fvg_css = input.color(color.new(#ff0008, 70), 'Bearish FVG'
  , group = 'Fair Value Gaps')

fvg_extend = input.int(1, "Extend FVG"
  , minval = 0
  , group = 'Fair Value Gaps'
  , tooltip = EXTEND_FVG)

//----------------------------------------}
//Previous day/week high/low
//----------------------------------------{
//Daily
show_pdhl = input(false, 'Daily'
  , inline = 'daily'
  , group = 'Highs & Lows MTF')

pdhl_style = input.string('⎯⎯⎯', ''
  , options = ['⎯⎯⎯', '----', '····']
  , inline = 'daily'
  , group = 'Highs & Lows MTF')

pdhl_css = input(#2157f3, ''
  , inline = 'daily'
  , group = 'Highs & Lows MTF')

//Weekly
show_pwhl = input(false, 'Weekly'
  , inline = 'weekly'
  , group = 'Highs & Lows MTF')

pwhl_style = input.string('⎯⎯⎯', ''
  , options = ['⎯⎯⎯', '----', '····']
  , inline = 'weekly'
  , group = 'Highs & Lows MTF')

pwhl_css = input(#2157f3, ''
  , inline = 'weekly'
  , group = 'Highs & Lows MTF')

//Monthly
show_pmhl = input(false, 'Monthly'
  , inline = 'monthly'
  , group = 'Highs & Lows MTF')

pmhl_style = input.string('⎯⎯⎯', ''
  , options = ['⎯⎯⎯', '----', '····']
  , inline = 'monthly'
  , group = 'Highs & Lows MTF')

pmhl_css = input(#2157f3, ''
  , inline = 'monthly'
  , group = 'Highs & Lows MTF')

//----------------------------------------}
//Premium/Discount zones
//----------------------------------------{
show_sd = input(false, 'Premium/Discount Zones'
  , group = 'Premium & Discount Zones'
  , tooltip = PED_ZONES)

premium_css = input.color(#f23645, 'Premium Zone'
  , group = 'Premium & Discount Zones')

eq_css = input.color(#b2b5be, 'Equilibrium Zone'
  , group = 'Premium & Discount Zones')

discount_css = input.color(#089981, 'Discount Zone'
  , group = 'Premium & Discount Zones')

//-----------------------------------------------------------------------------}
//Functions
//-----------------------------------------------------------------------------{
n = bar_index

atr = ta.atr(200)
cmean_range = ta.cum(high - low) / n

//HL Output function
hl() => [high, low]

//Get ohlc values function
get_ohlc()=> [close[1], open[1], high, low, high[2], low[2]]

//Display Structure function
display_Structure(x, y, txt, css, dashed, down, lbl_size)=>
    structure_line = line.new(x, y, n, y
      , color = css
      , style = dashed ? line.style_dashed : line.style_solid)

    structure_lbl = label.new(int(math.avg(x, n)), y, txt
      , color = TRANSP_CSS
      , textcolor = css
      , style = down ? label.style_label_down : label.style_label_up
      , size = lbl_size)

    if mode == 'Present'
        line.delete(structure_line[1])
        label.delete(structure_lbl[1])

//Swings detection/measurements
swings(len)=>
    var os = 0
    
    upper = ta.highest(len)
    lower = ta.lowest(len)

    os := high[len] > upper ? 0 : low[len] < lower ? 1 : os[1]

    top = os == 0 and os[1] != 0 ? high[len] : 0
    btm = os == 1 and os[1] != 1 ? low[len] : 0

    [top, btm]

//Order block coordinates function
ob_coord(use_max, loc, target_top, target_btm, target_left, target_type)=>
    min = 99999999.
    max = 0.
    idx = 1

    ob_threshold = ob_filter == 'Atr' ? atr : cmean_range 

    //Search for highest/lowest high within the structure interval and get range
    if use_max
        for i = 1 to (n - loc)-1
            if (high[i] - low[i]) < ob_threshold[i] * 2
                max := math.max(high[i], max)
                min := max == high[i] ? low[i] : min
                idx := max == high[i] ? i : idx
    else
        for i = 1 to (n - loc)-1
            if (high[i] - low[i]) < ob_threshold[i] * 2
                min := math.min(low[i], min)
                max := min == low[i] ? high[i] : max
                idx := min == low[i] ? i : idx

    array.unshift(target_top, max)
    array.unshift(target_btm, min)
    array.unshift(target_left, time[idx])
    array.unshift(target_type, use_max ? -1 : 1)

//Set order blocks
display_ob(boxes, target_top, target_btm, target_left, target_type, show_last, swing, size)=>
    for i = 0 to math.min(show_last-1, size-1)
        get_box = array.get(boxes, i)

        box.set_lefttop(get_box, array.get(target_left, i), array.get(target_top, i))
        box.set_rightbottom(get_box, array.get(target_left, i), array.get(target_btm, i))
        box.set_extend(get_box, extend.right)

        color css = na
        
        if swing 
            if style == 'Monochrome'
                css := array.get(target_type, i) == 1 ? color.new(#b2b5be, 80) : color.new(#5d606b, 80)
                border_css = array.get(target_type, i) == 1 ? #b2b5be : #5d606b
                box.set_border_color(get_box, border_css)
            else
                css := array.get(target_type, i) == 1 ? bull_ob_css : bear_ob_css
                box.set_border_color(get_box, css)

            box.set_bgcolor(get_box, css)
        else
            if style == 'Monochrome'
                css := array.get(target_type, i) == 1 ? color.new(#b2b5be, 80) : color.new(#5d606b, 80)
            else
                css := array.get(target_type, i) == 1 ? ibull_ob_css : ibear_ob_css
            
            box.set_border_color(get_box, css)
            box.set_bgcolor(get_box, css)
        
//Line Style function
get_line_style(style) =>
    out = switch style
        '⎯⎯⎯'  => line.style_solid
        '----' => line.style_dashed
        '····' => line.style_dotted

//Set line/labels function for previous high/lows
phl(h, l, tf, css)=>
    var line high_line = line.new(na,na,na,na
      , xloc = xloc.bar_time
      , color = css
      , style = get_line_style(pdhl_style))

    var label high_lbl = label.new(na,na
      , xloc = xloc.bar_time
      , text = str.format('P{0}H', tf)
      , color = TRANSP_CSS
      , textcolor = css
      , size = size.small
      , style = label.style_label_left)

    var line low_line = line.new(na,na,na,na
      , xloc = xloc.bar_time
      , color = css
      , style = get_line_style(pdhl_style))

    var label low_lbl = label.new(na,na
      , xloc = xloc.bar_time
      , text = str.format('P{0}L', tf)
      , color = TRANSP_CSS
      , textcolor = css
      , size = size.small
      , style = label.style_label_left)

    hy = ta.valuewhen(h != h[1], h, 1)
    hx = ta.valuewhen(h == high, time, 1)

    ly = ta.valuewhen(l != l[1], l, 1)
    lx = ta.valuewhen(l == low, time, 1)

    if barstate.islast
        ext = time + (time - time[1])*20

        //High
        line.set_xy1(high_line, hx, hy)
        line.set_xy2(high_line, ext, hy)

        label.set_xy(high_lbl, ext, hy)

        //Low
        line.set_xy1(low_line, lx, ly)
        line.set_xy2(low_line, ext, ly)

        label.set_xy(low_lbl, ext, ly)

//-----------------------------------------------------------------------------}
//Global variables
//-----------------------------------------------------------------------------{
var trend = 0, var itrend = 0

var top_y = 0., var top_x = 0
var btm_y = 0., var btm_x = 0

var itop_y = 0., var itop_x = 0
var ibtm_y = 0., var ibtm_x = 0

var trail_up = high, var trail_dn = low
var trail_up_x = 0,  var trail_dn_x = 0

var top_cross = true,  var btm_cross = true
var itop_cross = true, var ibtm_cross = true

var txt_top = '',  var txt_btm = ''

//Alerts
bull_choch_alert = false 
bull_bos_alert   = false 

bear_choch_alert = false 
bear_bos_alert   = false 

bull_ichoch_alert = false 
bull_ibos_alert   = false 

bear_ichoch_alert = false 
bear_ibos_alert   = false 

bull_iob_break = false 
bear_iob_break = false

bull_ob_break = false 
bear_ob_break = false

eqh_alert = false 
eql_alert = false 

//Structure colors
var bull_css = style == 'Monochrome' ? #b2b5be 
  : swing_bull_css

var bear_css = style == 'Monochrome' ? #b2b5be 
  : swing_bear_css

var ibull_css = style == 'Monochrome' ? #b2b5be 
  : swing_ibull_css

var ibear_css = style == 'Monochrome' ? #b2b5be 
  : swing_ibear_css

//Labels size
var internal_structure_lbl_size = internal_structure_size == 'Tiny' 
  ? size.tiny 
  : internal_structure_size == 'Small' 
  ? size.small 
  : size.normal 

var swing_structure_lbl_size = swing_structure_size == 'Tiny' 
  ? size.tiny 
  : swing_structure_size == 'Small' 
  ? size.small 
  : size.normal 

var eqhl_lbl_size = eq_size == 'Tiny' 
  ? size.tiny 
  : eq_size == 'Small' 
  ? size.small
  : size.normal 

//Swings
[top, btm] = swings(length)

[itop, ibtm] = swings(5)

//-----------------------------------------------------------------------------}
//Pivot High
//-----------------------------------------------------------------------------{
var line extend_top = na

var label extend_top_lbl = label.new(na, na
  , color = TRANSP_CSS
  , textcolor = bear_css
  , style = label.style_label_down
  , size = size.tiny)

if top
    top_cross := true
    txt_top := top > top_y ? 'HH' : 'LH'

    if show_swings
        top_lbl = label.new(n-length, top, txt_top
          , color = TRANSP_CSS
          , textcolor = bear_css
          , style = label.style_label_down
          , size = swing_structure_lbl_size)

        if mode == 'Present'
            label.delete(top_lbl[1])

    //Extend recent top to last bar
    line.delete(extend_top[1])
    extend_top := line.new(n-length, top, n, top
      , color = bear_css)

    top_y := top
    top_x := n - length

    trail_up := top
    trail_up_x := n - length

if itop
    itop_cross := true

    itop_y := itop
    itop_x := n - 5

//Trailing maximum
trail_up := math.max(high, trail_up)
trail_up_x := trail_up == high ? n : trail_up_x

//Set top extension label/line
if barstate.islast and show_hl_swings
    line.set_xy1(extend_top, trail_up_x, trail_up)
    line.set_xy2(extend_top, n + 20, trail_up)

    label.set_x(extend_top_lbl, n + 20)
    label.set_y(extend_top_lbl, trail_up)
    label.set_text(extend_top_lbl, trend < 0 ? 'Strong High' : 'Weak High')

//-----------------------------------------------------------------------------}
//Pivot Low
//-----------------------------------------------------------------------------{
var line extend_btm = na 

var label extend_btm_lbl = label.new(na, na
  , color = TRANSP_CSS
  , textcolor = bull_css
  , style = label.style_label_up
  , size = size.tiny)

if btm
    btm_cross := true
    txt_btm := btm < btm_y ? 'LL' : 'HL'
    
    if show_swings
        btm_lbl = label.new(n - length, btm, txt_btm
          , color = TRANSP_CSS
          , textcolor = bull_css
          , style = label.style_label_up
          , size = swing_structure_lbl_size)

        if mode == 'Present'
            label.delete(btm_lbl[1])
    
    //Extend recent btm to last bar
    line.delete(extend_btm[1])
    extend_btm := line.new(n - length, btm, n, btm
      , color = bull_css)

    btm_y := btm
    btm_x := n-length

    trail_dn := btm
    trail_dn_x := n-length

if ibtm
    ibtm_cross := true

    ibtm_y := ibtm
    ibtm_x := n - 5

//Trailing minimum
trail_dn := math.min(low, trail_dn)
trail_dn_x := trail_dn == low ? n : trail_dn_x

//Set btm extension label/line
if barstate.islast and show_hl_swings
    line.set_xy1(extend_btm, trail_dn_x, trail_dn)
    line.set_xy2(extend_btm, n + 20, trail_dn)

    label.set_x(extend_btm_lbl, n + 20)
    label.set_y(extend_btm_lbl, trail_dn)
    label.set_text(extend_btm_lbl, trend > 0 ? 'Strong Low' : 'Weak Low')

//-----------------------------------------------------------------------------}
//Order Blocks Arrays
//-----------------------------------------------------------------------------{
var iob_top = array.new_float(0)
var iob_btm = array.new_float(0)
var iob_left = array.new_int(0)
var iob_type = array.new_int(0)

var ob_top = array.new_float(0)
var ob_btm = array.new_float(0)
var ob_left = array.new_int(0)
var ob_type = array.new_int(0)

//-----------------------------------------------------------------------------}
//Pivot High BOS/CHoCH
//-----------------------------------------------------------------------------{
//Filtering
var bull_concordant = true

if ifilter_confluence
    bull_concordant := high - math.max(close, open) > math.min(close, open - low)

//Detect internal bullish Structure
if ta.crossover(close, itop_y) and itop_cross and top_y != itop_y and bull_concordant
    bool choch = na
    
    if itrend < 0
        choch := true
        bull_ichoch_alert := true
    else 
        bull_ibos_alert := true
    
    txt = choch ? 'CHoCH' : 'BOS'

    if show_internals
        if show_ibull == 'All' or (show_ibull == 'BOS' and not choch) or (show_ibull == 'CHoCH' and choch)
            display_Structure(itop_x, itop_y, txt, ibull_css, true, true, internal_structure_lbl_size)
    
    itop_cross := false
    itrend := 1
    
    //Internal Order Block
    if show_iob
        ob_coord(false, itop_x, iob_top, iob_btm, iob_left, iob_type)

//Detect bullish Structure
if ta.crossover(close, top_y) and top_cross
    bool choch = na
    
    if trend < 0
        choch := true
        bull_choch_alert := true
    else 
        bull_bos_alert := true

    txt = choch ? 'CHoCH' : 'BOS'
    
    if show_Structure
        if show_bull == 'All' or (show_bull == 'BOS' and not choch) or (show_bull == 'CHoCH' and choch)
            display_Structure(top_x, top_y, txt, bull_css, false, true, swing_structure_lbl_size)
    
    //Order Block
    if show_ob
        ob_coord(false, top_x, ob_top, ob_btm, ob_left, ob_type)

    top_cross := false
    trend := 1

//-----------------------------------------------------------------------------}
//Pivot Low BOS/CHoCH
//-----------------------------------------------------------------------------{
var bear_concordant = true

if ifilter_confluence
    bear_concordant := high - math.max(close, open) < math.min(close, open - low)

//Detect internal bearish Structure
if ta.crossunder(close, ibtm_y) and ibtm_cross and btm_y != ibtm_y and bear_concordant
    bool choch = false
    
    if itrend > 0
        choch := true
        bear_ichoch_alert := true
    else 
        bear_ibos_alert := true
    
    txt = choch ? 'CHoCH' : 'BOS'

    if show_internals
        if show_ibear == 'All' or (show_ibear == 'BOS' and not choch) or (show_ibear == 'CHoCH' and choch)
            display_Structure(ibtm_x, ibtm_y, txt, ibear_css, true, false, internal_structure_lbl_size)
    
    ibtm_cross := false
    itrend := -1
    
    //Internal Order Block
    if show_iob
        ob_coord(true, ibtm_x, iob_top, iob_btm, iob_left, iob_type)

//Detect bearish Structure
if ta.crossunder(close, btm_y) and btm_cross
    bool choch = na
    
    if trend > 0
        choch := true
        bear_choch_alert := true
    else 
        bear_bos_alert := true

    txt = choch ? 'CHoCH' : 'BOS'
    
    if show_Structure
        if show_bear == 'All' or (show_bear == 'BOS' and not choch) or (show_bear == 'CHoCH' and choch)
            display_Structure(btm_x, btm_y, txt, bear_css, false, false, swing_structure_lbl_size)
    
    //Order Block
    if show_ob
        ob_coord(true, btm_x, ob_top, ob_btm, ob_left, ob_type)

    btm_cross := false
    trend := -1

//-----------------------------------------------------------------------------}
//Order Blocks
//-----------------------------------------------------------------------------{
//Set order blocks
var iob_boxes = array.new_box(0)
var ob_boxes = array.new_box(0)

//Delete internal order blocks box coordinates if top/bottom is broken
for element in iob_type
    index = array.indexof(iob_type, element)

    if close < array.get(iob_btm, index) and element == 1
        array.remove(iob_top, index) 
        array.remove(iob_btm, index) 
        array.remove(iob_left, index) 
        array.remove(iob_type, index)
        bull_iob_break := true

    else if close > array.get(iob_top, index) and element == -1
        array.remove(iob_top, index) 
        array.remove(iob_btm, index)
        array.remove(iob_left, index) 
        array.remove(iob_type, index)
        bear_iob_break := true

//Delete internal order blocks box coordinates if top/bottom is broken
for element in ob_type
    index = array.indexof(ob_type, element)

    if close < array.get(ob_btm, index) and element == 1
        array.remove(ob_top, index) 
        array.remove(ob_btm, index) 
        array.remove(ob_left, index) 
        array.remove(ob_type, index)
        bull_ob_break := true

    else if close > array.get(ob_top, index) and element == -1
        array.remove(ob_top, index) 
        array.remove(ob_btm, index)
        array.remove(ob_left, index) 
        array.remove(ob_type, index)
        bear_ob_break := true

iob_size = array.size(iob_type)
ob_size = array.size(ob_type)

if barstate.isfirst
    if show_iob
        for i = 0 to iob_showlast-1
            array.push(iob_boxes, box.new(na,na,na,na, xloc = xloc.bar_time))
    if show_ob
        for i = 0 to ob_showlast-1
            array.push(ob_boxes, box.new(na,na,na,na, xloc = xloc.bar_time))

if iob_size > 0
    if barstate.islast
        display_ob(iob_boxes, iob_top, iob_btm, iob_left, iob_type, iob_showlast, false, iob_size)

if ob_size > 0
    if barstate.islast
        display_ob(ob_boxes, ob_top, ob_btm, ob_left, ob_type, ob_showlast, true, ob_size)

//-----------------------------------------------------------------------------}
//EQH/EQL
//-----------------------------------------------------------------------------{
var eq_prev_top = 0.
var eq_top_x = 0

var eq_prev_btm = 0.
var eq_btm_x = 0

if show_eq
    eq_top = ta.pivothigh(eq_len, eq_len)
    eq_btm = ta.pivotlow(eq_len, eq_len)

    if eq_top 
        max = math.max(eq_top, eq_prev_top)
        min = math.min(eq_top, eq_prev_top)
        
        if max < min + atr * eq_threshold
            eqh_line = line.new(eq_top_x, eq_prev_top, n-eq_len, eq_top
              , color = bear_css
              , style = line.style_dotted)

            eqh_lbl = label.new(int(math.avg(n-eq_len, eq_top_x)), eq_top, 'EQH'
              , color = #00000000
              , textcolor = bear_css
              , style = label.style_label_down
              , size = eqhl_lbl_size)

            if mode == 'Present'
                line.delete(eqh_line[1])
                label.delete(eqh_lbl[1])
            
            eqh_alert := true

        eq_prev_top := eq_top
        eq_top_x := n-eq_len

    if eq_btm 
        max = math.max(eq_btm, eq_prev_btm)
        min = math.min(eq_btm, eq_prev_btm)
        
        if min > max - atr * eq_threshold
            eql_line = line.new(eq_btm_x, eq_prev_btm, n-eq_len, eq_btm
              , color = bull_css
              , style = line.style_dotted)

            eql_lbl = label.new(int(math.avg(n-eq_len, eq_btm_x)), eq_btm, 'EQL'
              , color = #00000000
              , textcolor = bull_css
              , style = label.style_label_up
              , size = eqhl_lbl_size)

            eql_alert := true

            if mode == 'Present'
                line.delete(eql_line[1])
                label.delete(eql_lbl[1])

        eq_prev_btm := eq_btm
        eq_btm_x := n-eq_len

//-----------------------------------------------------------------------------}
//Fair Value Gaps
//-----------------------------------------------------------------------------{
var bullish_fvg_max = array.new_box(0)
var bullish_fvg_min = array.new_box(0)

var bearish_fvg_max = array.new_box(0)
var bearish_fvg_min = array.new_box(0)

float bullish_fvg_avg = na
float bearish_fvg_avg = na

bullish_fvg_cnd = false
bearish_fvg_cnd = false

[src_c1, src_o1, src_h, src_l, src_h2, src_l2] =
  request.security(syminfo.tickerid, fvg_tf, get_ohlc())

if show_fvg
    delta_per = (src_c1 - src_o1) / src_o1 * 100

    change_tf = timeframe.change(fvg_tf)

    threshold = fvg_auto ? ta.cum(math.abs(change_tf ? delta_per : 0)) / n * 2 
      : 0

    //FVG conditions
    bullish_fvg_cnd := src_l > src_h2
      and src_c1 > src_h2 
      and delta_per > threshold
      and change_tf

    bearish_fvg_cnd := src_h < src_l2 
      and src_c1 < src_l2 
      and -delta_per > threshold
      and change_tf

    //FVG Areas
    if bullish_fvg_cnd
        array.unshift(bullish_fvg_max, box.new(n-1, src_l, n + fvg_extend, math.avg(src_l, src_h2)
          , border_color = bull_fvg_css
          , bgcolor = bull_fvg_css))
        
        array.unshift(bullish_fvg_min, box.new(n-1, math.avg(src_l, src_h2), n + fvg_extend, src_h2
          , border_color = bull_fvg_css
          , bgcolor = bull_fvg_css))
    
    if bearish_fvg_cnd
        array.unshift(bearish_fvg_max, box.new(n-1, src_h, n + fvg_extend, math.avg(src_h, src_l2)
          , border_color = bear_fvg_css
          , bgcolor = bear_fvg_css))
        
        array.unshift(bearish_fvg_min, box.new(n-1, math.avg(src_h, src_l2), n + fvg_extend, src_l2
          , border_color = bear_fvg_css
          , bgcolor = bear_fvg_css))

    for bx in bullish_fvg_min
        if low < box.get_bottom(bx)
            box.delete(bx)
            box.delete(array.get(bullish_fvg_max, array.indexof(bullish_fvg_min, bx)))
    
    for bx in bearish_fvg_max
        if high > box.get_top(bx)
            box.delete(bx)
            box.delete(array.get(bearish_fvg_min, array.indexof(bearish_fvg_max, bx)))

//-----------------------------------------------------------------------------}
//Previous day/week high/lows
//-----------------------------------------------------------------------------{
//Daily high/low
[pdh, pdl] = request.security(syminfo.tickerid, 'D', hl()
  , lookahead = barmerge.lookahead_on)

//Weekly high/low
[pwh, pwl] = request.security(syminfo.tickerid, 'W', hl()
  , lookahead = barmerge.lookahead_on)

//Monthly high/low
[pmh, pml] = request.security(syminfo.tickerid, 'M', hl()
  , lookahead = barmerge.lookahead_on)

//Display Daily
if show_pdhl
    phl(pdh, pdl, 'D', pdhl_css)

//Display Weekly
if show_pwhl
    phl(pwh, pwl, 'W', pwhl_css)
    
//Display Monthly
if show_pmhl
    phl(pmh, pml, 'M', pmhl_css)

//-----------------------------------------------------------------------------}
//Premium/Discount/Equilibrium zones
//-----------------------------------------------------------------------------{
var premium = box.new(na, na, na, na
  , bgcolor = color.new(premium_css, 80)
  , border_color = na)

var premium_lbl = label.new(na, na
  , text = 'Premium'
  , color = TRANSP_CSS
  , textcolor = premium_css
  , style = label.style_label_down
  , size = size.small)

var eq = box.new(na, na, na, na
  , bgcolor = color.rgb(120, 123, 134, 80)
  , border_color = na)

var eq_lbl = label.new(na, na
  , text = 'Equilibrium'
  , color = TRANSP_CSS
  , textcolor = eq_css
  , style = label.style_label_left
  , size = size.small)

var discount = box.new(na, na, na, na
  , bgcolor = color.new(discount_css, 80)
  , border_color = na)

var discount_lbl = label.new(na, na
  , text = 'Discount'
  , color = TRANSP_CSS
  , textcolor = discount_css
  , style = label.style_label_up
  , size = size.small)

//Show Premium/Discount Areas
if barstate.islast and show_sd
    avg = math.avg(trail_up, trail_dn)

    box.set_lefttop(premium, math.max(top_x, btm_x), trail_up)
    box.set_rightbottom(premium, n, .95 * trail_up + .05 * trail_dn)

    label.set_xy(premium_lbl, int(math.avg(math.max(top_x, btm_x), n)), trail_up)

    box.set_lefttop(eq, math.max(top_x, btm_x), .525 * trail_up + .475*trail_dn)
    box.set_rightbottom(eq, n, .525 * trail_dn + .475 * trail_up)

    label.set_xy(eq_lbl, n, avg)
    
    box.set_lefttop(discount, math.max(top_x, btm_x), .95 * trail_dn + .05 * trail_up)
    box.set_rightbottom(discount, n, trail_dn)
    label.set_xy(discount_lbl, int(math.avg(math.max(top_x, btm_x), n)), trail_dn)

//-----------------------------------------------------------------------------}
//Trend
//-----------------------------------------------------------------------------{
var color trend_css = na

if show_trend
    if style == 'Colored'
        trend_css := itrend == 1 ? bull_css : bear_css
    else if style == 'Monochrome'
        trend_css := itrend == 1 ? #b2b5be : #5d606b

plotcandle(open, high, low, close
  , color = trend_css
  , wickcolor = trend_css
  , bordercolor = trend_css
  , editable = false)

//-----------------------------------------------------------------------------}
//Alerts
//-----------------------------------------------------------------------------{
//Internal Structure
alertcondition(bull_ibos_alert, 'Internal Bullish BOS', 'Internal Bullish BOS formed')
alertcondition(bull_ichoch_alert, 'Internal Bullish CHoCH', 'Internal Bullish CHoCH formed')

alertcondition(bear_ibos_alert, 'Internal Bearish BOS', 'Internal Bearish BOS formed')
alertcondition(bear_ichoch_alert, 'Internal Bearish CHoCH', 'Internal Bearish CHoCH formed')

//Swing Structure
alertcondition(bull_bos_alert, 'Bullish BOS', 'Internal Bullish BOS formed')
alertcondition(bull_choch_alert, 'Bullish CHoCH', 'Internal Bullish CHoCH formed')

alertcondition(bear_bos_alert, 'Bearish BOS', 'Bearish BOS formed')
alertcondition(bear_choch_alert, 'Bearish CHoCH', 'Bearish CHoCH formed')

//order Blocks
alertcondition(bull_iob_break, 'Bullish Internal OB Breakout', 'Price broke bullish internal OB')
alertcondition(bear_iob_break, 'Bearish Internal OB Breakout', 'Price broke bearish internal OB')

alertcondition(bull_ob_break, 'Bullish Swing OB Breakout', 'Price broke bullish swing OB')
alertcondition(bear_ob_break, 'Bearish Swing OB Breakout', 'Price broke bearish swing OB')

//EQH/EQL
alertcondition(eqh_alert, 'Equal Highs', 'Equal highs detected')
alertcondition(eql_alert, 'Equal Lows', 'Equal lows detected')

//FVG
alertcondition(bullish_fvg_cnd, 'Bullish FVG', 'Bullish FVG formed')
alertcondition(bearish_fvg_cnd, 'Bearish FVG', 'Bearish FVG formed')

//-----------------------------------------------------------------------------}