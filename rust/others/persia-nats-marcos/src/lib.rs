use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::{Attribute, FnArg, Ident, ImplItem, PatType, Receiver, ReturnType};

#[allow(dead_code)]
struct NatsSubject {
    attrs: Vec<Attribute>,
    ident: Ident,
    arg: PatType,
    receiver: Receiver,
    output: ReturnType,
}

impl NatsSubject {
    fn subject_ident(&self) -> Ident {
        quote::format_ident!("{}", self.ident)
    }

    fn subscribe_subject_ident(&self) -> Ident {
        quote::format_ident!("subscribe_{}", self.ident)
    }

    fn publish_subject_ident(&self) -> Ident {
        quote::format_ident!("publish_{}", self.ident)
    }

    fn req_type(&self) -> TokenStream2 {
        let arg_type = &self.arg.ty;
        quote::quote! {
            #arg_type
        }
    }

    fn resp_type(&self) -> TokenStream2 {
        let output = &self.output;
        let output = match output {
            ReturnType::Default => syn::Type::Verbatim(quote::quote! {()}),
            ReturnType::Type(_, t) => (**t).clone(),
        };
        quote::quote! {
            #output
        }
    }

    fn subscription(&self, service_type_name: String) -> TokenStream2 {
        let subject_ident = self.subject_ident();
        let subject_string = subject_ident.to_string();
        let subscribe_ident = self.subscribe_subject_ident();
        let req_type = self.req_type();
        let spawn_task = quote::quote! {
            let instance = self.inner.clone();
            let subscription = nats_client.subscribe(&subject).await?;
            persia_libs::tokio::spawn(async move {
                while let Some(msg) = subscription.next().await {
                    let arg: Result<#req_type, _> = tokio::task::block_in_place(|| {
                        ::persia_speedy::Readable::read_from_buffer(&msg.data)
                    });
                    if let Ok(a) = arg {
                        let result = instance.#subject_ident(a).await;
                        let resp = tokio::task::block_in_place(|| {
                            result.write_to_vec().unwrap()
                        });
                        let result = msg.respond(resp).await;
                        if result.is_err() {
                            tracing::error!("failed to respond for {}", #subject_string);
                        }
                    }
                    else {
                        tracing::error!("failed to decode msg from nats of {}", #subject_string);
                    }
                }
            });
        };
        quote::quote! {
            pub async fn #subscribe_ident(&self) -> Result<(), NatsError> {
                let nats_client = self.nats_client.clone();
                let replica_info = PersiaReplicaInfo::get().expect("failed to get replica_info");
                let subject = nats_client.get_subject(
                    #service_type_name,
                    #subject_string,
                    Some(replica_info.replica_index),
                );

                #spawn_task

                let subject = nats_client.get_subject(
                    #service_type_name,
                    #subject_string,
                    None,
                );

                #spawn_task

                Ok(())
            }
        }
    }

    fn publishing(&self, service_type_name: String) -> TokenStream2 {
        let subject_ident = self.subject_ident();
        let subject_string = subject_ident.to_string();
        let publish_ident = self.publish_subject_ident();
        let resp_type = self.resp_type();
        let req_type = self.req_type();

        quote::quote! {
            pub async fn #publish_ident(
                &self,
                message: &#req_type,
                dst_index: Option<usize>,
            ) -> Result<#resp_type, NatsError> {
                let nats_client = self.nats_client.clone();
                let subject = nats_client.get_subject(
                    #service_type_name,
                    #subject_string,
                    dst_index,
                );

                let msg = tokio::task::block_in_place(|| {
                    message.write_to_vec().unwrap()
                });

                let resp_bytes: Vec<u8>  = self.nats_client.request(&subject, &msg).await?;
                let response = tokio::task::block_in_place(|| {
                    ::persia_speedy::Readable::read_from_buffer(&resp_bytes)
                });
                match response {
                    Ok(resp) => Ok(resp),
                    Err(_) => Err(NatsError::DecodeError),
                }
            }
        }
    }
}

struct NatsService {
    subjects: Vec<NatsSubject>,
    service_type_name: String,
}

impl NatsService {
    fn responder_impl(&self) -> TokenStream2 {
        let service_ident = quote::format_ident!("{}", self.service_type_name);
        let responder_ident = quote::format_ident!("{}Responder", self.service_type_name);
        let subscriptions: Vec<_> = self
            .subjects
            .iter()
            .map(|x| x.subscription(self.service_type_name.clone()))
            .collect();
        let subscribe_subjects: Vec<_> = self
            .subjects
            .iter()
            .map(|x| {
                let subscribe_ident = x.subscribe_subject_ident();
                quote::quote! {
                    if let Err(e) = self.#subscribe_ident().await {
                        panic!("failed to subsrcibe due to {:?}", e);
                    }
                }
            })
            .collect();

        quote::quote! {
            pub struct #responder_ident {
                inner: #service_ident,
                nats_client: NatsClient,
            }

            impl #responder_ident {
                pub async fn new(service: #service_ident) -> Self {
                    let instance = Self {
                        inner: service,
                        nats_client: NatsClient::get().await,
                    };
                    instance.spawn_subscriptions().await.expect("failed to spawn nats subscriptions");
                    instance
                }

                #( #subscriptions )*

                pub async fn spawn_subscriptions(&self) -> Result<(), NatsError> {
                    #( #subscribe_subjects )*
                    Ok(())
                }
            }
        }
    }

    fn publisher_impl(&self) -> TokenStream2 {
        let publisher_ident = quote::format_ident!("{}Publisher", self.service_type_name);
        let publish: Vec<_> = self
            .subjects
            .iter()
            .map(|x| x.publishing(self.service_type_name.clone()))
            .collect();

        quote::quote! {
            pub struct #publisher_ident {
                nats_client: NatsClient
            }

            impl #publisher_ident {
                pub async fn new() -> Self {
                    Self {
                        nats_client: NatsClient::get().await,
                    }
                }
                #( #publish )*
            }
        }
    }
}

#[proc_macro_attribute]
pub fn service(_attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(tokens as syn::ItemImpl);
    let service_type_name = item.self_ty.clone().into_token_stream().to_string();
    let subjects: Vec<NatsSubject> = item
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Method(m) => Some(m),
            _ => None,
        })
        .map(|m| {
            let mut args = vec![];
            let mut receiver = None;
            for arg in &m.sig.inputs {
                match arg {
                    FnArg::Receiver(captures) => {
                        if captures.mutability.is_some() {
                            panic!("should be &self");
                        } else if receiver.is_some() {
                            panic!("duplicated self");
                        } else {
                            receiver = Some(captures.clone());
                        }
                    }
                    FnArg::Typed(captures) => match *captures.pat {
                        syn::Pat::Ident(_) => args.push(captures.clone()),
                        _ => panic!("patterns aren't allowd in RPC args"),
                    },
                }
            }
            assert_eq!(args.len(), 1, "only support single argument");
            NatsSubject {
                attrs: m.attrs.clone(),
                ident: m.sig.ident.clone(),
                arg: args[0].clone(),
                receiver: receiver.unwrap(),
                output: m.sig.output.clone(),
            }
        })
        .collect();

    let service = NatsService {
        subjects,
        service_type_name,
    };

    let responder_impl = service.responder_impl();
    let publisher_impl = service.publisher_impl();

    (quote::quote! {
        #item

        #responder_impl

        #publisher_impl

    })
    .into()
}
