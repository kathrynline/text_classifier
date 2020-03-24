#--------------------------------------------------
# SAMPLE OF TB ACTIVITIES FROM DRC AND SEN GRANTS 
# TO BE USED TO HAND-CLASSIFY ACTIVITIES AND RUN A MACHINE LEARNING MODEL. 
#-------------------------------------------------

#Pull Senegal and DRC final budgets 
sen <- readRDS("J:/Project/Evaluation/GF/resource_tracking/_gf_files_gos/sen/prepped_data/final_budgets.rds")
cod <- readRDS("J:/Project/Evaluation/GF/resource_tracking/_gf_files_gos/cod/prepped_data/final_budgets.rds")

all = rbind(sen, cod)

tb = unique(all[grant_disease=="tb", .(orig_module, orig_intervention, code, activity_description, grant_disease, grant, grant_period)])
nrow(tb) #539 

#Set seed, and generate a random number. 
set.seed(nrow(tb)) #539
tb[, rand:=runif(nrow(tb))]
tb = tb[order(rand)]

tb[1:100, SAMPLED:=TRUE]

write.xlsx(tb, "J:/Project/Evaluation/GF/resource_tracking/modular_framework_mapping/nlp/tb_training_data.xlsx")

