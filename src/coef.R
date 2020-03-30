df <- read.csv(file = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/src/coefs/data/officefloor.csv')
y = df['Buggy']
X = within(df, rm('Buggy'))
print(X)
#glad
#model <- glm(Buggy~avg_NPRM+max_NPRM+NRev+NStmt+total_ModifiedLOC+max_NOM+NFix,data = df, family=binomial(link="logit"))
#selenium
#model <- glm(Buggy~total_FANOUT+NRev+max_FANOUT+NFix+max_NPM+max_ModifiedLOC+total_CBO+total_ModifiedLOC+avg_RFC+NStmt+total_NPM,data = df, family=binomial(link="logit"))
#oscarmcmaster
#model <- glm(Buggy~NFix+CL+max_ModifiedLOC+avg_FANOUT+max_NOM+avg_AddedLOC+max_CBO+avg_DIT+avg_NOC+total_DeletedLOC,data = df, family=binomial(link="logit"))
#synecdoche
#model <- glm(Buggy~avg_NPM+total_FANOUT+avg_DIT+max_ModifiedLOC+max_NPM+avg_FANIN+max_RFC+avg_AddedLOC+total_NPM+avg_ModifiedLOC+total_NPRM+avg_DIT.1+max_CC+avg_NOC,data = df, family=binomial(link="logit"))
#officefloor
model <- glm(Buggy~max_NPRM+NFix+total_NIV+total_NPRM+total_CBO+max_DeletedLOC+MNL+avg_NOC+max_NIV+avg_FANIN+total_ModifiedLOC+total_NPM,data = df, family=binomial(link="logit"))
print(summary(model))