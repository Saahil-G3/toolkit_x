from .surv_tools import (
    get_median_survival, 
    get_km_binary, 
    plot_km_binary,
    get_hazard_binary
)

class BinSurv:
    
    def __init__(self, df, column_information=None):
        self.df = df
        self.column_information = column_information
    
    def get_km_by_threshold_binary(
        self,
        t2e_var,
        threshold,
        continuousVarName,
        plot_title=None,
        labels=None,
        save_path=None,
        **args,
        ):
    
        
        self.surv =  self.df[[t2e_var['time'], t2e_var['event'], continuousVarName]].copy()

        categoriesVarName = f"{continuousVarName}_groups"
        self.surv[categoriesVarName] = (self.surv[continuousVarName] >= threshold).astype(int)
        self.surv.drop(columns=[continuousVarName], inplace=True)
    
        group_0 = self.surv[self.surv[categoriesVarName]==0]
        group_1 = self.surv[self.surv[categoriesVarName]==1]
    
        hazard_ratio, hazard_ratio_ci = get_hazard_binary(self.surv, t2e_var, categoriesVarName)
        km_binary_dict = get_km_binary(group_0, group_1, t2e_var)

        if labels is None:
            labels = [f"< {round(threshold,2)}", f">={round(threshold,2)}"]
            
        if plot_title is None:
            plot_title = f"{t2e_var['name']} {continuousVarName}"

        plot_km_binary(km_binary_dict=km_binary_dict, 
                       plot_title=plot_title, 
                       labels=labels ,
                       hazard_ratio=hazard_ratio, 
                       hazard_ratio_ci=hazard_ratio_ci,
                       save_path=save_path,
                       **args
                      )
