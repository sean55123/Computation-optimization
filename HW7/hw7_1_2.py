from pyomo.environ import *

model = ConcreteModel()

# Fixed costs ($/hr)
model.F_i  = 10  # Process I
model.F_ii = 15  # Process II
model.F_iii= 20  # Process III

# Variable costs ($/ton of raw material)
model.V_i  = 2.5   # Process I
model.V_ii = 4.0   # Process II
model.V_iii= 5.5   # Process III

# Raw material prices ($/ton)
model.PriceA = 5.0  
model.PriceB = 9.5  

model.max_C = 15
model.max_highC = 10
model.Price_highC = 18
model.Price_lowC = 15

# Conversion factors
model.convAtoB = 0.90  # process I
model.convBtoC_ii = 0.82  # process II
model.convBtoC_iii= 0.95  # process III

# Maximum supply of A (tons/hr)
model.max_A = 16

# If installed or not
model.pI   = Var(domain=Binary) 
model.pII  = Var(domain=Binary)
model.pIII = Var(domain=Binary)

# Flow of A into process I
model.Fa = Var(domain=NonNegativeReals)
# B purchased
model.bp = Var(domain=NonNegativeReals)
# B fed to Process II
model.bII = Var(domain=NonNegativeReals)
# B fed to Process III
model.bIII = Var(domain=NonNegativeReals)
# C produced in high price
model.C_high = Var(domain=NonNegativeReals)
# C produecd in low price
model.C_low = Var(domain=NonNegativeReals)

# 1) Exclusive of process 2 and process 3
def exclusive_rule(model):
    return model.pII + model.pIII <= 1
model.exclusive_constraint = Constraint(rule=exclusive_rule)

# 2) Capacity constraint of A
#    Big M equals to the maximum amount of A
def process_I_capacity(model):
    return model.Fa <= 16 * model.pI
model.capacity_I = Constraint(rule=exclusive_rule)

# 3) B fed to Process II only if II is built
#    Big M equals to the amount of flow to b as all produced by process 2
def process_II_capacity(model):
    return model.bII <= 13 * model.pII
model.capacity_II = Constraint(rule=process_II_capacity)

# 4) B fed to Process III only if III is built
#    Big M equals to the amount of flow to b as all produced by process 1
def process_III_capacity(model):
    return model.bIII <= 11 * model.pIII
model.capacity_III = Constraint(rule=process_III_capacity)

# 5) Limit A supply
def limit_A(model):
    return model.Fa <= model.max_A
model.limit_A = Constraint(rule=limit_A)

# 6) B balance:
#    B total = B from process I + B purchased
#             => bII + bIII <= convAtoB * aI + bPurchase
def B_balance(model):
    return (model.bII + model.bIII 
            <= model.convAtoB*model.Fa + model.bp)
model.limit_B = Constraint(rule=B_balance)

# 7) Limitation of c:
def limit_C(model):
    return model.C_low + model.C_high <= model.max_C
model.limit_C = Constraint(rule=limit_C)

# 8) Limitation of high c:
def limit_highC(model):
    return model.C_high <= model.max_highC

# 9) C balance
def balance_C(model):
    return model.C_high + model.C_low == model.convBtoC_ii*model.bII + model.convBtoC_iii*model.bIII
model.C_balance = Constraint(rule=balance_C)

fixed_cost = model.pI*model.F_i + model.pII*model.F_ii + model.pIII*model.F_iii
var_cost   = model.Fa*model.V_i + model.bII*model.V_ii + model.bIII*model.V_iii
mat_cost   = model.Fa*model.PriceA + model.bp*model.PriceB

revenue = model.C_high*model.Price_highC + model.C_low*model.Price_lowC
model.Profit = Objective(expr=revenue - (fixed_cost + var_cost + mat_cost), sense=maximize)

solver = SolverFactory('gurobi') 
results = solver.solve(model, tee=True)

print(f"Is process II built: {model.pII.value}.")
print(f"Is process III built: {model.pIII.value}")
print(f"{model.Fa.value:.3f} ton/hr of A is from process I, and {model.bp.value:.3f} ton/hr of B is purchased")
print(f"{model.C_high.value:.3f} ton/hr amount of C is produced in high price")
print(f"{model.C_low.value:.3f} ton/hr amount of C is produced in low price")
print(f"The final profit is {round(value(model.Profit), 3)} kUSD/hr")