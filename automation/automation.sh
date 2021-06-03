#!/bin/bash

python automation/finish_status.py

# u to h
source=ucf101
target=hmdb51
full=true
use_cdan=false
use_attention=false
method=no
use_i3d=true
mode=flow
python main.py --source ucf101
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

# u to h cdan part
use_cdan=true
use_attention=false
method=no
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

# h to u
source=hmdb51
target=ucf101
use_attention=false
method=no
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

#h to u cdan part
use_cdan=true
use_attention=false
method=no
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode


#### small
# u to h
source=ucf101
target=hmdb51
full=false
use_cdan=false
use_attention=false
method=no
use_i3d=true
mode=flow
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

# u to h cdan part
use_cdan=true
use_attention=false
method=no
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

# h to u
source=hmdb51
target=ucf101
use_attention=false
method=no
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

#h to u cdan part
use_cdan=true
use_attention=false
method=no
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=false
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

use_attention=true
method=path_gen
python main.py --source $source --target $target --full $full --use_cdan $use_cdan --use_attention $use_attention --method $method --use_i3d $use_i3d --mode $mode

python ./automation/finish_status.py