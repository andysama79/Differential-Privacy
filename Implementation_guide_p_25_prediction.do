use "private_data_by_cells.dta", clear
local omega_dist normal
	gen theta_g = .     //estimate of interest
	gen SE_theta_g = .  //SE of estimate of interest
	gen theta_g_d = . 	//estimate obtained when adding one observation
	gen LS_g = .		//local sensitivity
	
	levelsof cell, local(cells)  //save list of cells in local
	foreach g of local cells { 
		qui reg kid_rank parent_rank  if cell == `g'
		replace theta_g = _b[parent_rank]*0.25 + _b[_cons] if cell == `g'
		lincom _b[parent_rank]*0.25 + _b[_cons]
		replace SE_theta_g = r(se) if cell == `g'
			count
			local additional_obs = r(N) + 1
			set obs `additional_obs'
			replace cell = `g' if  _n == `additional_obs'
			replace LS_g = 0 if cell == `g' 
			forvalues i = 0/3 {
			
				replace parent_rank = floor(`i'/2) if _n==`additional_obs'
				replace kid_rank    = mod(`i',2)   if _n==`additional_obs'

				qui reg kid_rank parent_rank if cell==`g'
				replace theta_g_d = _b[parent_rank]*0.25 + _b[_cons] if cell == `g'
				replace LS_g = abs(theta_g_d - theta_g) if abs(theta_g_d - theta_g) > LS_g & cell == `g'
				
				}
		
		drop if  _n==`additional_obs'
	}

	bys cell: gen N_g = _N
	gen N_g_LS_g = N_g * LS_g
	egen chi = max(N_g_LS_g)
	collapse theta_g SE_theta_g N_g chi, by(cell)
	set seed 419
	forval epsilon = 1(1)10 {
		local draws = 500
		forval d=1(1)`draws' {
		
		if "`omega_dist'"=="normal"		gen omega_`d' = rnormal(0, 1)       		// Normal
		if "`omega_dist'"=="laplace"	gen omega_`d' = rlaplace(0, 1/sqrt(2)) 		// Laplace
		gen noise_infused_theta_g_`d' = theta_g + sqrt(2)*(chi / (`epsilon' * N_g)) * omega_`d'
		gen diff_true_noise_`d' = (noise_infused_theta_g _`d' - theta_g) ^ 2
		}
	egen MSE_eps_`epsilon' = rowmean(diff_true_noise_*)
	
	drop omega_*  noise_infused_theta_g_* diff_true_noise_*
	}
	preserve
	collapse MSE_eps_*
	gen id=1
	reshape long MSE_eps_, i(id) j(epsilon)
	twoway (connected MSE_eps_ epsilon), ///
	xtitle("Epsilon") ytitle( "Mean Squared Error (Average)" " ")
	restore
	
	drop MSE_eps_*
	local epsilon = 4
	
	set seed 5711
	if "`omega_dist'"=="normal"		gen omega = rnormal(0, 1)       		// Normal
	if "`omega_dist'"=="laplace"	gen omega = rlaplace(0, 1/sqrt(2)) 		// Laplace
	
	gen noise_infused_theta_g = theta_g + sqrt(2)*(chi / (`epsilon' * N_g))*omega
	gen SE_noise_infused_theta_g = sqrt(SE_theta_g^2 + 2*((chi / (`epsilon' * N_g))^2))
	gen noise_infused_N_g = N_g + sqrt(2)*(omega / `epsilon')
	gen SD_noise_g = sqrt(2) * (chi / (`epsilon' * N_g))
	sum SD_noise_g
	sum noise_infused_theta_g
	local total_variance = `r(Var)'
	gen Var_noise_g = SD_noise_g ^ 2 
	sum Var_noise_g
	local noise_var = `r(mean)'

	local share_noise_variance = `noise_var' / `total_variance'
	dis %4.3f `share_noise_variance'
	drop theta_g SE_theta_g N_g omega chi Var_noise_g
	export excel using "example_cell_public_estimates_p25.xlsx", replace firstr(var)
