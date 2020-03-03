library(dplyr)
library(quanteda)
library(caret)
library(parallelSVM)
library(fastrtext)
library(tictoc) # Needed for timings
library(reshape2)
library(tensorflow)
library(keras)
library(doMC) # Needed for parallel grid search
registerDoMC(cores = 4)

# Load the datasets
cleaned_hm <- read.csv("data/cleaned_hm.csv", stringsAsFactors = FALSE)
demographics <- read.csv("data/demographic.csv", stringsAsFactors = FALSE)

# Merge the datasets by "wid"
df <- cleaned_hm %>% inner_join(demographics, by = "wid")

# Select only the happy moment text and gender columns
gender_df <- df %>% select('cleaned_hm', 'gender')

# Select only texts from male and females
gender_df <- gender_df[(gender_df$gender == 'm') | (gender_df$gender == 'f'), ]
unique(gender_df$gender)

# Keep only those lines that have no null values
gender_df <- gender_df[complete.cases(gender_df), ]

# Remove exact duplicates
gender_df <- gender_df %>% dplyr::distinct(.keep_all = TRUE)

# Create a new column that contains the binarization the gender column:
#   Male: 1
#   Female: 0
gender_df$gender_bin <- ifelse(gender_df$gender == 'm', 1, 0)
unique(gender_df$gender_bin)

# Gender distribution
counts <- table(gender_df$gender)
barplot(counts)
females <- as.numeric(length(which(gender_df[,"gender_bin"] == 0)))
females
males <- as.numeric(length(which(gender_df[,"gender_bin"] == 1)))
males
print(paste0("Males: ", round(males/(females+males)*100), "%"))
print(paste0("Females: ", round(females/(females+males)*100), "%"))

# Text Cleaning - Create corpus
gender_df$cleaned_hm <- gsub("_","",gender_df$cleaned_hm)
gender_df$cleaned_hm <- gsub("'","",gender_df$cleaned_hm)
corpus <- corpus(gender_df, text_field = "cleaned_hm")
token <- tokens(corpus, what = "word",
              remove_url = TRUE,
              remove_punct = TRUE,
              remove_numbers = TRUE,
              remove_twitter = TRUE, 
              remove_hyphens = TRUE,
              remove_symbols = TRUE, 
              remove_separators = TRUE)
token <- tokens_remove(token, "^[0-9]+", valuetype="regex")
token <- tokens_remove(token, stopwords("english"))
token <- tokens_wordstem(token, language = "english")
token <- tokens_tolower(token)

# Create the document-feature matrix
data <- dfm(token)

# Convert it to using tfidf
data <- dfm_tfidf(data, scheme_tf = "prop")

# To reduce the dimension of the DFM, 
# we can remove the less frequent terms such that
# the sparsity is less than 0.99
data <- dfm_trim(data, sparsity=0.992, verbose = TRUE)

# Wordcloud
col <- sapply(seq(0.1, 1, 0.1), function(x) adjustcolor("#1F78B4", x))
textplot_wordcloud(data, adjust = 0.5, 
                   random_order = FALSE, color = col, rotation = FALSE)

# Convert the dfm to dataframe
data <- convert(data, to="data.frame")
data$document <- NULL
colnames(data) <- gsub("'","a", colnames(data), fixed=TRUE)
data$y <- factor(gender_df$gender_bin)

# Remove rows with the same words in the same order and
# that have the same gender
data <- data %>% dplyr::distinct(.keep_all = TRUE)

y <- data$y

# Create the train/test split
index <- createDataPartition(y, times = 1, p = 0.7, list = FALSE)
train <- data[index, ]
test <- data[-index, ]
nrow(train)
nrow(test)

y_train <- train$y
train$y <- NULL
y_test <- test$y
test$y <- NULL

# Free up some memory
rm(cleaned_hm, corpus, demographics, df)
gc()

## Algorithms ##
# SVM
modSVM <- parallelSVM(x=train, y=y_train, numberCores = 4)
predictedTest <- predict(modSVM, test)
confusionMatrix(y_test, predictedTest)

# Grid search SVM
set.seed(123)
trctrl <- trainControl(method = "cv", number = 5,
                       verboseIter = TRUE, allowParallel = TRUE)
svm_Linear <- train(x=train, y=y_train, method = "svmRadial",
                    trControl = trctrl,
                    tuneLength = 3,
                    verbose = TRUE)

svm_Linear
plot(svm_Linear)

saveRDS(svm_Linear, "svm_model.rds")
svm_Linear <- readRDS("svm_model.rds")

predictedSVM <- predict(svm_Linear, test)
cmSVM <- confusionMatrix(y_test, predictedSVM)
cmSVM

# GLM
modelGLM <- train(x=train, y=y_train, method = "glm",
             trControl=trainControl(method = "none",
                                    verboseIter = TRUE,
                                    allowParallel = TRUE),
             tuneLength=1)

saveRDS(modelGLM, "glm_model.rds")
modelGLM <- readRDS("glm_model.rds")

pred <- predict(modelGLM, test)
cmGLM <- confusionMatrix(y_test, pred)
cmGLM

# Stochastic Gradient Boosting
trctrl <- trainControl(method = "none", verboseIter = TRUE, allowParallel = TRUE)
gbm <- train(x=train, y=y_train, method = "gbm",
                  trControl=trctrl,
                  verbose = TRUE)

predgbm <- predict(gbm, test)
confusionMatrix(y_test, predgbm)

# Grid-search GBM
grid <- expand.grid(shrinkage = c(0.1,0.4,0.7),
                    interaction.depth = c(3,5),
                    n.trees = c(10,20,50),
                    n.minobsinnode=10)
trctrl <- trainControl(method = "cv", number = 5,
                       verboseIter = TRUE, allowParallel = TRUE)
gbm_best <- train(x=train, y=y_train, method = "gbm",
                  trControl = trctrl,
                  tuneGrid = grid,
                  verbose = TRUE)

plot(gbm_best)
summary(gbm_best)

saveRDS(gbm_best, "gbm_model.rds")
gbm_best <- readRDS("gbm_model.rds")

predictedTestGBM <- predict(gbm_best, test)
cmGBM <- confusionMatrix(y_test, predictedTestGBM)
cmGBM

# FastText
# We need to join each word of a token element into a single sentence
token_list <- as.list(token)
myFunc <- function(x) {
  paste(unlist(token_list[x]), collapse=' ')
}
sentences <- mclapply(1:length(token_list), myFunc, mc.cores = 6)

# We split again the data
#index <- createDataPartition(y, times = 1, p = 0.7, list = FALSE)
train_ft <- sentences[index]
test_ft <- sentences[-index]

# FastText works with files saved on the disk
# Thus, we save a temp file for training and one for testing
tmp_file_model <- tempfile()

train_labels <- paste0("__label__", y[index])
train_to_write <- paste(train_labels, train_ft)
train_tmp_file_txt <- tempfile()
writeLines(text = train_to_write, con = train_tmp_file_txt)

test_labels <- paste0("__label__", y[-index])
test_to_write <- paste(test_labels, test_ft)

# Compile and save the model
execute(commands = c("supervised", "-input", train_tmp_file_txt,
                     "-output", tmp_file_model,
                     "-minn", 2, "-maxn", 5, "-lr", 0.8,
                     "-wordNgrams", 2, "-epoch", 10, "-verbose", 10))

# Load back the model and see the test accuracy
model <- load_model(tmp_file_model)
predictions <- predict(model, sentences = test_to_write)
mean(names(unlist(predictions)) == gender_df$gender_bin[-index])

# Grid search FastText
learning_rates <- seq(0.1,1,0.1)
epochs <- c(1,5,10,15)
counter <- 1

gs <- expand.grid(learning_rates, epochs)
gs[,"accuracy"] <- NA
colnames(gs) <- c("lr", "epoch", "accuracy")

bestModel <- NULL
for (i in 1:nrow(gs))
{
  # Start timer
  tic(paste0("Training model ",counter," (learning rate: ",
             gs$lr[i],", epochs: ",gs$epoch[i],")"))
  print(paste0("Training model ",counter," (learning rate: ",
               gs$lr[i],", epochs: ",gs$epoch[i],")"))
  
  # Train model
  execute(commands = c("supervised", "-input", train_tmp_file_txt,
                       "-output", tmp_file_model, "-minn", 2, "-maxn", 5,
                       "-lr", gs$lr[i], "-epoch", gs$epoch[i], "-verbose", 10))
  
  # Stop timer
  toc(log=TRUE)
  
  # Load model and see test accuracy
  model <- load_model(tmp_file_model)
  predictions <- predict(model, sentences = test_to_write)
  
  # Save prediction to the results dataframe
  gs$accuracy[i] <- mean(names(unlist(predictions)) == y[-index])
  
  counter <- counter + 1
  
  # Try to save the best model
  if (i==1 || gs$accuracy[i] > as.numeric(gs[which.max(gs$accuracy), "accuracy"]))
  {
    bestModel <- model
  }
  
  # Clean memory
  rm(model)
  gc()
}

# See the train times
log.lst <- tic.log(format = FALSE)
timings <- unlist(lapply(log.lst, function(x) x$toc - x$tic))
timings

# Get the model with highest accuracy (row index of dataframe)
gs
rowInd <- as.numeric(which(gs[,"accuracy"] ==
                             gs[which.max(gs$accuracy), "accuracy"]))
timings[rowInd]

# Plot the grid search results
p <- ggplot(gs, aes(lr, accuracy, group=factor(epoch))) +
  geom_line(aes(color=factor(epoch)))
p <- p + ggtitle("FastText grid search") + 
  theme(plot.title = element_text(hjust = 0.5))
p <- p + labs(x = "learning rate", colour = "epochs")
p

# Stacked Auto-Encoder
dnn_best <- train(x=data.matrix(train),y=y_train, method = "dnn",
                  trControl=trainControl(method="cv", number = 5,
                                         verboseIter = TRUE,
                                         allowParallel = TRUE),
                  tuneGrid=expand.grid(layer1=c(1,2,3),
                                       layer2=c(1,2,3),
                                       layer3=c(1,2,3),
                                       visible_dropout=0,
                                       hidden_dropout=0))

plot(dnn_best)
saveRDS(dnn_best, "ae_model.rds")
dnn_best <- readRDS("ae_model.rds")

predictedTestdnn <- predict(dnn_best,data.matrix(test))
cm = table(y_test,predictedTestdnn)
print(cm)
acc_ae <- sum(diag(cm))/sum(cm)
print(acc_ae)

# Keras NN
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu') %>%    
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'softmax')
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = 'accuracy'
)

tic.clear()
tic.clearlog()
tic("Keras training")

history <- model %>% fit(
  as.matrix(train), to_categorical(y_train), 
  epochs = 15, batch_size = 128, 
  validation_split = 0.2
)
toc(log=TRUE)
log.lst <- tic.log(format = FALSE)
timings_keras <- unlist(lapply(log.lst, function(x) x$toc - x$tic))
timings_keras <- timings_keras/15 # DIVIDED BY N OF EPOCHS

plot(history)
model %>% save_model_hdf5("keras.h5")
model <- load_model_hdf5("keras.h5")
keras_acc <- model %>% evaluate(as.matrix(test), to_categorical(y_test))
keras_acc

# Keras CNN
modelCNN <- keras_model_sequential()
modelCNN %>%
  layer_embedding(197, 1, input_length = 197) %>%
  layer_dropout(0.2) %>%
  
  layer_conv_1d(
    100, 3, activation = "relu"
  ) %>%
  
  # Apply max pooling:
  layer_global_max_pooling_1d() %>%
  layer_dropout(0.2) %>%
  
  layer_flatten() %>%
  
  layer_dense(256, activation = "relu") %>%
  layer_dropout(0.4) %>%
  layer_dense(128, activation = "relu") %>%
  layer_dropout(0.3) %>%
  
  layer_dense(2, activation = "sigmoid")

modelCNN %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = 'accuracy'
)

tic.clear()
tic.clearlog()
tic("CNN training")

historyCNN <- modelCNN %>% fit(
  as.matrix(train), to_categorical(y_train), 
  epochs = 15, batch_size = 128, 
  validation_split = 0.2
)
toc(log=TRUE)
log.lst <- tic.log(format = FALSE)
timings_CNN <- unlist(lapply(log.lst, function(x) x$toc - x$tic))
timings_CNN <- timings_CNN/15 # DIVIDED BY N OF EPOCHS

plot(historyCNN)
modelCNN %>% save_model_hdf5("cnn.h5")
modelCNN <- load_model_hdf5("cnn.h5")
cnn_acc <- modelCNN %>% evaluate(as.matrix(test), to_categorical(y_test))
cnn_acc

## Final results table
accs <- c(as.numeric(cmGLM$overall["Accuracy"]),
          as.numeric(cmSVM$overall["Accuracy"]),
          as.numeric(cmGBM$overall["Accuracy"]),
          as.numeric(keras_acc$acc),
          as.numeric(acc_ae),
          as.numeric(cnn_acc$acc),
          as.numeric(gs[which.max(gs$accuracy), "accuracy"]))

train_times <- c(as.numeric(modelGLM$times$final["elapsed"]),
                 as.numeric(svm_Linear$times$final["elapsed"]),
                 as.numeric(gbm_best$times$final["elapsed"]),
                 as.numeric(timings_keras[1]),
                 as.numeric(dnn_best$times$final["elapsed"]),
                 as.numeric(timings_CNN[1]),
                 as.numeric(timings[rowInd]))

final_results <- data.frame(accs, train_times, stringsAsFactors = FALSE)
rownames(final_results) <- c("GLM", "SVM", "Gradient Boosting",
                             "Neural Network", "Stacked Auto-Encoder",
                             "CNN", "FastText")
colnames(final_results) <- c("Accuracy", "Training Time")
final_results
