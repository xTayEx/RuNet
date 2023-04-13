#include <gtest/gtest.h>
#include <runet/global/global.h>

//TEST(demo_suit, demo_test) {
//  EXPECT_EQ(1 + 10, 11);
//}
//
//TEST(test_demo, vector_test) {
//  std::vector<int> v1 {1, 2, 3};
//  std::vector<int> v2 {1, 4, 3};
//  EXPECT_EQ(v1, v2);
//}
//
int main() {
  RuNet::init_context();
  testing::InitGoogleTest();
  int test_status = RUN_ALL_TESTS();
  RuNet::destroy_context();
  return test_status;
}