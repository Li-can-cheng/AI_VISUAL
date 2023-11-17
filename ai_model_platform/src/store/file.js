import { defineStore } from 'pinia'

// 你可以对 `defineStore()` 的返回值进行任意命名，但最好使用 store 的名字，同时以 `use` 开头且以 `Store` 结尾。(比如 `useUserStore`，`useCartStore`，`useProductStore`)
// 第一个参数是你的应用中 Store 的唯一 ID。
export const useFileStore = defineStore('file', {
  // 其他配置...
  state: () => {
    return {
      // 所有这些属性都将自动推断出它们的类型
      files: {

      },
      curent_file: {
        name: "",
        file: null,
        dom: null,
        type: ""
      }
    }
  },
  getters: {
    // get_file(state,name){
    //     return state.files[name];
    // }
  },
  actions: {
    set_file(name, file) {
      this.files[name] = file;
    },
    get_file(name) {
      return this.files[name];
    },
    get_current_file(){
      return this.current_file;
    },
    set_current_file(current_file){
      this.current_file = current_file;
    }
  }
})