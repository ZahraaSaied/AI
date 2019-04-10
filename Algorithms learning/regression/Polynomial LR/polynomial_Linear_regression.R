dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

#linear model
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
#polynomial model
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

#visualising Linear model 
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red")+
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = "red")+
  ggtitle("Linear model")+
  xlab("Levels")+
  ylab("Salary")

#visualising polynomial model
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red")+
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
                colour = "blue")+
  ggtitle("Polynomial model")+
  xlab("Levels")+
  ylab("Salary")
  
#linear model prediction
y_pred = predict(lin_reg, data.frame(Level = 6.5))
  
#polynomial model prediction
y_pred2 = predict(poly_reg, data.frame(Level = 6.5, 
                                       Level2 = 6.5^2, 
                                       Level3 = 6.5^3, 
                                       Level4 = 6.5^4))






  
  